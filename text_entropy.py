from typing import List, Optional, Dict, Tuple
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity  # Updated import
from transformers import AutoTokenizer, AutoModel
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from umap import UMAP

class TextEntropyCalculator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the entropy calculator with a specific embedding model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings for each token in the text."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the last hidden state
            embeddings = outputs.last_hidden_state.cpu().numpy()[0]
        return embeddings

    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarities between token embeddings."""
        return cosine_similarity(embeddings)

    def _similarity_to_probability(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Convert similarity matrix to probability distribution using softmax."""
        exp_sim = np.exp(similarity_matrix)
        return exp_sim / exp_sim.sum(axis=1, keepdims=True)

    def _compute_entropy(self, prob_distribution: np.ndarray) -> float:
        """Compute entropy from probability distribution."""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        entropy = -np.sum(prob_distribution * np.log2(prob_distribution + epsilon))
        return entropy

    def calculate_entropy(self, text: str) -> float:
        """Calculate the entropy of the given text."""
        # Get embeddings for each token
        embeddings = self._get_embeddings(text)
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(embeddings)
        
        # Convert to probability distribution
        prob_distribution = self._similarity_to_probability(similarity_matrix)
        
        # Calculate entropy
        entropy = self._compute_entropy(prob_distribution)
        
        # Normalize by number of tokens
        normalized_entropy = entropy / len(embeddings)
        return normalized_entropy

class TopicEntropyCalculator:
    def __init__(self, min_topic_size: int = 2):
        """
        Initialize the topic entropy calculator with BERTopic.
        
        Args:
            min_topic_size: Minimum size of topics
        """
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Configure UMAP with parameters suitable for small texts
        umap_model = UMAP(
            n_neighbors=2,  # Reduced from default
            n_components=2,  # Reduced dimensions
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            min_topic_size=min_topic_size,
            umap_model=umap_model,
            verbose=False,
            nr_topics='auto'  # Let BERTopic decide the number of topics
        )
        
    def _segment_text(self, text: str, window_size: int = 2) -> List[str]:
        """
        Segment text into overlapping windows of sentences.
        
        Args:
            text: Input text
            window_size: Number of sentences per segment
            
        Returns:
            List of text segments
        """
        # Better sentence splitting
        sentences = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                # Split by period but keep sentences that end with numbers (e.g., "2.5")
                parts = [s.strip() for s in line.split('.')]
                sentences.extend([s for s in parts if s])
        
        # Ensure we have enough sentences
        if len(sentences) < window_size:
            return sentences
        
        # Create overlapping windows
        segments = []
        for i in range(len(sentences) - window_size + 1):
            segment = ' '.join(sentences[i:i + window_size])
            if len(segment.split()) >= 5:  # Only keep segments with at least 5 words
                segments.append(segment)
        
        return segments

    def _get_topic_labels(self, text: str) -> List[str]:
        """
        Get topic labels for text segments using BERTopic.
        
        Args:
            text: Input text
            
        Returns:
            List of topic labels for each segment
        """
        # Segment the text
        segments = self._segment_text(text)
        
        # Debug print
        print(f"Number of segments: {len(segments)}")
        print("Segments:")
        for i, seg in enumerate(segments):
            print(f"{i+1}: {seg[:100]}...")
        
        # Check if we have enough segments
        if len(segments) < 3:
            print("Warning: Not enough segments for topic modeling")
            return ["default_topic"] * len(segments)
        
        try:
            # Fit and transform the topic model
            topics, probs = self.topic_model.fit_transform(segments)
            
            # Get topic labels
            topic_labels = []
            for topic_idx in topics:
                if topic_idx == -1:  # BERTopic assigns -1 to outliers
                    topic_labels.append("misc")
                else:
                    try:
                        # Get the most representative words for this topic
                        topic_words = self.topic_model.get_topic(topic_idx)
                        if topic_words and len(topic_words) > 0:
                            # Use first word as label
                            topic_labels.append(topic_words[0][0])
                        else:
                            topic_labels.append("misc")
                    except:
                        topic_labels.append("misc")
            
            # Debug print
            print("\nDetected topics:")
            for i, label in enumerate(topic_labels):
                print(f"Segment {i+1}: {label}")
            
            return topic_labels
            
        except Exception as e:
            print(f"Warning: Topic modeling failed ({str(e)}). Using default topic assignment.")
            return ["default_topic"] * len(segments)

    def _get_topic_segments(self, topic_labels: List[str]) -> Dict[str, List[Tuple[int, int]]]:
        """
        Find continuous stretches of text for each topic.
        
        Args:
            topic_labels: List of topic labels
            
        Returns:
            Dictionary mapping topics to lists of (start, end) positions
        """
        segments = defaultdict(list)
        current_topic = None
        start_pos = 0
        
        for i, topic in enumerate(topic_labels):
            if topic != current_topic:
                if current_topic is not None:
                    segments[current_topic].append((start_pos, i))
                current_topic = topic
                start_pos = i
        
        # Don't forget the last segment
        if current_topic is not None:
            segments[current_topic].append((start_pos, len(topic_labels)))
            
        return segments

    def calculate_topic_entropy(self, text: str) -> Tuple[float, Dict[str, float]]:
        """
        Calculate topic entropy for a text using automatic topic detection.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (mean_entropy, topic_entropies)
            where topic_entropies is a dict mapping topics to their individual entropies
        """
        # Get topic labels using BERTopic
        topic_labels = self._get_topic_labels(text)
        
        # Get segments for each topic
        topic_segments = self._get_topic_segments(topic_labels)
        
        # Calculate entropy for each topic
        topic_entropies = {}
        
        for topic, segments in topic_segments.items():
            # Calculate total length of text for this topic
            total_length = sum(end - start for start, end in segments)
            
            # Calculate probability for each continuous stretch
            probabilities = [(end - start) / total_length for start, end in segments]
            
            # Calculate entropy using formula from paper:
            # S(α) = -∑(p(αi) * log(p(αi)))
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            topic_entropies[topic] = entropy
        
        # Calculate mean entropy across topics
        mean_entropy = sum(topic_entropies.values()) / len(topic_entropies) if topic_entropies else 0
        
        return mean_entropy, topic_entropies

def calculate_text_entropy(text: str, model_name: Optional[str] = None) -> float:
    """Convenience function to calculate text entropy."""
    calculator = TextEntropyCalculator(model_name) if model_name else TextEntropyCalculator()
    return calculator.calculate_entropy(text)

def calculate_text_topic_entropy(text: str) -> float:
    """Convenience function to calculate topic entropy."""
    calculator = TopicEntropyCalculator()
    mean_entropy, _ = calculator.calculate_topic_entropy(text)
    return mean_entropy

# Example usage
if __name__ == "__main__":
    # Example text with more clearly separated topics
    text = """
    The quantum mechanics of black holes intertwine with the socioeconomic impacts of climate change. As we delve into the intricacies of string theory, we must also consider the role of artificial intelligence in modern agriculture. The Renaissance art movement shares surprising parallels with the development of cryptocurrency blockchain technology. Meanwhile, the mating habits of deep-sea creatures offer insights into urban planning and sustainable architecture. Ancient Mayan astronomy techniques could revolutionize our approach to space exploration, while simultaneously informing best practices in digital marketing. The philosophy of existentialism has unexpected applications in machine learning algorithms, just as the principles of interpretive dance can enhance our understanding of geopolitical conflicts. Lastly, the fermentation process in artisanal cheese-making holds the key to unraveling the mysteries of dark matter in the universe.
    """
    
    calculator = TopicEntropyCalculator()
    mean_entropy, topic_entropies = calculator.calculate_topic_entropy(text)
    
    print(f"\nMean Topic Entropy: {mean_entropy:.3f}")
    print("\nEntropy by topic:")
    for topic, entropy in topic_entropies.items():
        print(f"{topic}: {entropy:.3f}")
