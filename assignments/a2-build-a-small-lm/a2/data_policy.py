"""Part 4: exact-document deduplication with an auditable representative rule."""
import unicodedata


def normalize_text(text):
    """NFC, CRLF/CR -> LF, strip outer whitespace. Preserve case and internal whitespace."""
    # YOUR CODE HERE
    raise NotImplementedError


def deduplicate(documents):
    """Input: list of {'id': nonempty str, 'text': str, ...} with unique IDs.

    Return (retained_documents, removed_to_retained_id). Preserve the original
    dictionary/text for the lexicographically smallest ID per normalized group;
    sort retained documents by ID. Empty input returns ([], {}). Reject blank
    normalized text, duplicate IDs, or invalid id/text fields with ValueError.
    Do not alter input. Related-but-distinct text must remain distinct.
    """
    # YOUR CODE HERE
    raise NotImplementedError
