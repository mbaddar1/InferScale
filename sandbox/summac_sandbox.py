import nltk
nltk.download('punkt_tab')
from summac.model_summac import SummaCConv
if __name__ == '__main__':
    model = SummaCConv(models=["vitc"], device="cpu")  # or "cuda"
    source_text = """Artificial intelligence (AI) is transforming industries across the globe. In healthcare, 
    AI-powered systems assist doctors in diagnosing diseases more accurately and efficiently. 
    In finance, machine learning models detect fraudulent transactions and improve investment strategies.   
    Meanwhile, in transportation, self-driving technologies are being developed to reduce accidents and enhance mobility.
     Despite these benefits, AI also raises concerns about job displacement, data privacy, and ethical decision-making. 
     As AI continues to evolve, it is crucial for policymakers, businesses, and individuals to work together to ensure 
     its responsible and beneficial use."""
    generated_summary = ("AI is transforming multiple industries like healthcare, finance, "
                         "and transportation by improving efficiency and decision-making, "
                         "but it also raises concerns about jobs, privacy, and ethics, "
                         "requiring responsible use.")
    score = model.score([source_text], [generated_summary])
    print(score)
