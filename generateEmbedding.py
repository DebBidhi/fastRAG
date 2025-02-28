from datasets import load_dataset

# Step 1: Load the SQuAD dataset
dataset = load_dataset("squad")

# Step 2: Extract unique contexts{set} from the dataset
data = [item["context"] for item in dataset["train"]]
texts = list(set(data))
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm



def batch_iterate(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

class EmbedData:

    def __init__(self, 
                 embed_model_name="nomic-ai/nomic-embed-text-v1.5",
                 batch_size=32):
        
        self.embed_model_name = embed_model_name
        self.embed_model = self._load_embed_model()
        self.batch_size = batch_size
        self.embeddings = []

    def _load_embed_model(self):
        embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name,
                                           trust_remote_code=True,
                                           cache_folder='./hf_cache')
        return embed_model

    
    def generate_embedding(self, context):
        return self.embed_model.get_text_embedding_batch(context)
    
    def embed(self, contexts):
        self.contexts = contexts
        
        total_batches = (len(contexts) + self.batch_size - 1) // self.batch_size

        with tqdm(total=total_batches, desc="Embedding data", unit="batch") as pbar:
            for batch_context in batch_iterate(contexts, self.batch_size):
                batch_embeddings = self.generate_embedding(batch_context)
                self.embeddings.extend(batch_embeddings)
                pbar.update(1)  


batch_size = 32

embeddata = EmbedData(batch_size=batch_size)

embeddata.embed(texts)

# Save embeddings and contexts to pickle file
import pickle

with open('embeddings_data.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddata.embeddings, 'contexts': embeddata.contexts}, f, protocol=pickle.HIGHEST_PROTOCOL)


