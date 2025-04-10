from transformers import pipeline

class TextGenerator:
    def __init__(self):
        # Initialize the text generation pipeline with GPT-2
        self.generator = pipeline("text-generation", model="gpt2", max_length=50)

    def generate_response(self, emotion, text_input):

        # Create a prompt for the language model
        prompt = f"I'm feeling {emotion}! {text_input} "

        # Generate the response
        generated = self.generator(prompt, num_return_sequences=1, max_length=50)[0]["generated_text"]
        return generated.strip()
  
# Instantiate the text generator
text_generator = TextGenerator()