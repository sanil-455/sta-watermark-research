import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Lightweight embedding model for fast semantic evaluation
EMBEDDING_MODEL_PATH = 'sentence-transformers/all-MiniLM-L6-v2'


class SemanticChecker:
    """
    Evaluates semantic preservation and edit distance between 
    original watermarked text and modified (attacked) text.
    """
    def __init__(self, model_name=EMBEDDING_MODEL_PATH, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[SemanticChecker] Loading similarity encoder on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return F.normalize(sentence_embeddings, p=2, dim=1)

    def compute_similarity(self, text_orig, text_attacked):
        emb_orig = self.get_embedding(text_orig)
        emb_attacked = self.get_embedding(text_attacked)
        similarity = torch.mm(emb_orig, emb_attacked.T).item()
        return float(similarity)

    @staticmethod
    def compute_token_edit_distance(tokens_orig, tokens_attacked):
        m, n = len(tokens_orig), len(tokens_attacked)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens_orig[i - 1] == tokens_attacked[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Deletion
                        dp[i][j - 1],      # Insertion
                        dp[i - 1][j - 1]   # Replacement
                    )
        
        edit_dist = dp[m][n]
        rel_distance = edit_dist / float(max(m, 1))
        return edit_dist, rel_distance


def main():
    print("=" * 60)
    print("TESTING SEMANTIC CHECKER MODULE")
    print("=" * 60)
    
    checker = SemanticChecker()

    t1 = "I love Fiona. I'll take you to Cincinnati with me - although it looks like she might be coming to us."
    t2 = "I love Fiona. I'll take you to Cincinnati with me - although it looks like she could be coming to us."

    sim_score = checker.compute_similarity(t1, t2)
    print(f"Original Text: {t1}")
    print(f"Attacked Text: {t2}")
    print(f"Semantic Cosine Similarity: {sim_score:.4f}")
    
    if sim_score >= 0.90:
        print("[✓ PASSED] High semantic preservation preserved (> 0.90)")
    else:
        print("[✗ FAILED] Severe semantic drift detected!")
    print("=" * 60)


if __name__ == '__main__':
    main()