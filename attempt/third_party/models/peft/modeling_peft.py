import torch
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import PromptTuningConfig, get_peft_model, PeftModel

# === Custom Attentive Prompt Embedding === #
class AttentivePromptEmbedding(torch.nn.Module):
    def __init__(self, model_dim, num_virtual_tokens, temperature=0.5):
        """
        Implements attention-based soft prompt selection on top of PEFT's prompt embeddings.
        """
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.temperature = temperature
        self.model_dim = model_dim

        # Attention mechanism parameters
        self.attn_logits = torch.nn.Parameter(torch.randn(num_virtual_tokens))

    def forward(self, prompt_embeddings):
        """
        Applies attention-based weighting to the prompt embeddings.
        """
        batch_size, num_tokens, hidden_dim = prompt_embeddings.shape

        # Sample attention weights using Relaxed Bernoulli
        attn_scores = RelaxedBernoulli(temperature=self.temperature, logits=self.attn_logits).rsample()
        attn_scores = F.softmax(attn_scores, dim=0)  # Normalize

        # Compute final prompt embeddings as a weighted sum of virtual token embeddings
        attentive_prompt = torch.einsum("i,bij->bj", attn_scores, prompt_embeddings)

        # Expand for batch
        attentive_prompt = attentive_prompt.unsqueeze(1).repeat(1, num_tokens, 1)  # Shape: (batch_size, num_virtual_tokens, model_dim)
        return attentive_prompt

# === Custom Model Class to Wrap Everything === #
class CustomModel(PeftModel):
    def __init__(self, base_model, peft_config, attn_prompt_embedding):
        super().__init__(base_model, peft_config)
        self.attentive_prompt_encoder = attn_prompt_embedding

    def forward(self, input_ids, inputs_embeds=None, attention_mask=None, **kwargs):
        # Step 1: Get virtual token embeddings from PromptEncoder
        adapter_name = list(self.peft_config.keys())[0]  # Get active adapter name
        indices = torch.arange(self.peft_config.num_virtual_tokens, dtype=torch.long, device=self.device)
        virtual_prompt_embeddings = self.prompt_encoder[adapter_name](indices)
        virtual_prompt_embeddings = virtual_prompt_embeddings.unsqueeze(0).expand(input_ids.size(0), -1, -1)

        # Step 2: Apply AttentivePromptEmbedding for refined embeddings
        attentive_embeddings = self.attentive_prompt_encoder(virtual_prompt_embeddings)

        # Step 3: Get input embeddings
        input_embeddings = self.base_model.shared(input_ids) if inputs_embeds is None else inputs_embeds

        # Step 4: Inject attentive embeddings into inputs
        inputs_embeds = torch.cat([attentive_embeddings, input_embeddings], dim=1)

        # Return the forward pass output
        return self.base_model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

# === Setup for Training === #
def setup_training():
    # Load pre-trained model and tokenizer
    model_name_or_path = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    # PEFT configuration
    peft_config = PromptTuningConfig(
        task_type="SEQ_2_SEQ_LM",
        num_virtual_tokens=10,  # Define number of soft prompt tokens
        tokenizer_name_or_path=model_name_or_path
    )

    # Initialize AttentivePromptEmbedding
    attn_prompt_embedding = AttentivePromptEmbedding(
        model_dim=base_model.config.d_model,
        num_virtual_tokens=peft_config.num_virtual_tokens,
    )

    # Initialize custom model with attentive prompt embedding
    model = CustomModel(base_model, peft_config, attn_prompt_embedding)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")  # Move model to GPU if available

    return model, tokenizer

# === Training Setup === #
def train_model(model, tokenizer):
    # Load dataset (GLUE MRPC)
    dataset = load_dataset("glue", "mrpc")
    
    # Preprocess the dataset for model input
    def preprocess_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding=True, max_length=128)

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # Setup data collator and training arguments
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model.base_model)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    return model

# === Inference Function (After Training) === #
def generate_output(model, tokenizer, input_text):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Ensure tensors are on the correct device

    # Forward pass with the trained model
    outputs = model.generate(input_ids=inputs["input_ids"], inputs_embeds=inputs["inputs_embeds"])

    # Decode and print output
    print("Generated Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# === Full Training and Inference Example === #
if __name__ == "__main__":
    # Setup model and tokenizer
    model, tokenizer = setup_training()

    # Train the model
    trained_model = train_model(model, tokenizer)

    # Inference with generated output
    input_text = "Translate English to French: Hello, how are you?"
    generate_output(trained_model, tokenizer, input_text)

