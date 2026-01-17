from typing import List, Tuple

class Chunker:
    """
    Splits text into overlapping chunks to handle large documents.
    """
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Recursively splits text using a list of separators to respect semantic boundaries.
        Separators: ["\n\n", "\n", " ", ""]
        """
        separators = ["\n\n", "\n", " ", ""]
        
        def split(text, separators):
            if len(text) <= self.chunk_size:
                return [text]

            if not separators:
                return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

            separator = separators[0]
            next_separators = separators[1:]

            if separator == "":
                return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

            items = text.split(separator)
            chunks = []
            current_chunk = []
            current_len = 0

            for item in items:
                # Check if adding this item exceeds the chunk size
                item_len = len(item)
                sep_len = len(separator) if current_chunk else 0
                
                if current_len + item_len + sep_len <= self.chunk_size:
                    current_chunk.append(item)
                    current_len += item_len + sep_len
                else:
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                        current_chunk = []
                        current_len = 0

                    if item_len > self.chunk_size:
                        chunks.extend(split(item, next_separators))
                    else:
                        current_chunk.append(item)
                        current_len = item_len

            if current_chunk:
                chunks.append(separator.join(current_chunk))

            return chunks

        return split(text, separators)

    def chunk_pair(self, ground_truth: str, candidate: str) -> List[Tuple[str, str]]:
        """
        Chunks both texts.
        
        WARN: This assumes texts are roughly aligned. If candidate is 
        half the length of GT (e.g., summary vs full text), simple slicing 
        will misalign them. 
        
        For this simplified implementation, we assume GT and Candidate are 
        comparable in length/structure (e.g., translation, paraphasing).
        """
        gt_chunks = self.chunk_text(ground_truth)
        cand_chunks = self.chunk_text(candidate)
        
        # Naive alignment: zip them.
        # Real-world solution requires semantic alignment (e.g. using embeddings).
        # We'll truncate to the shorter list for safety in this demo.
        min_len = min(len(gt_chunks), len(cand_chunks))
        return list(zip(gt_chunks[:min_len], cand_chunks[:min_len]))
