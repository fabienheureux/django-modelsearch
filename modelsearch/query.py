#
# Base classes
#


class SearchQuery:
    def __and__(self, other):
        return And([self, other])

    def __or__(self, other):
        return Or([self, other])

    def __invert__(self):
        return Not(self)

    def __repr__(self):
        raise NotImplementedError


#
# Basic query classes
#


class PlainText(SearchQuery):
    OPERATORS = ["and", "or"]
    DEFAULT_OPERATOR = "and"

    def __init__(
        self, query_string: str, operator: str = DEFAULT_OPERATOR, boost: float = 1
    ):
        self.query_string = query_string
        self.operator = operator.lower()
        if self.operator not in self.OPERATORS:
            raise ValueError("`operator` must be either 'or' or 'and'.")
        self.boost = boost

    def __repr__(self):
        return f"<PlainText {repr(self.query_string)} operator={repr(self.operator)} boost={repr(self.boost)}>"


class Phrase(SearchQuery):
    def __init__(self, query_string: str):
        self.query_string = query_string

    def __repr__(self):
        return f"<Phrase {repr(self.query_string)}>"


class Fuzzy(SearchQuery):
    OPERATORS = ["and", "or"]
    DEFAULT_OPERATOR = "or"

    def __init__(
        self,
        query_string: str,
        operator: str = DEFAULT_OPERATOR,
        unaccent: bool = False,
    ):
        self.query_string = query_string
        self.operator = operator.lower()
        self.unaccent = unaccent
        if self.operator not in self.OPERATORS:
            raise ValueError("`operator` must be either 'or' or 'and'.")

    def __repr__(self):
        return f"<Fuzzy {repr(self.query_string)} operator={repr(self.operator)}>"


class Semantic(SearchQuery):
    """A semantic (vector) search query.

    Generates an embedding for the query string and finds results
    whose stored embeddings are most similar (cosine similarity).

    Args:
        query_string: The text to search for semantically.
        threshold: Minimum cosine similarity (0.0 to 1.0) for a result
            to be included. Defaults to the backend's configured threshold.
        unaccent: If True, normalize accents before generating the embedding.
    """

    def __init__(
        self,
        query_string: str,
        threshold: float | None = None,
        unaccent: bool = False,
    ):
        self.query_string = query_string
        self.threshold = threshold
        self.unaccent = unaccent

    def __repr__(self):
        return f"<Semantic {repr(self.query_string)} threshold={self.threshold}>"


class Hybrid(SearchQuery):
    """A hybrid search query combining fuzzy and semantic search.

    Runs both fuzzy (trigram) and semantic (vector) search, then combines
    the scores using the configured weights. This gives the best of both
    worlds: exact/near-exact matches from fuzzy search, and conceptually
    similar results from semantic search.

    Args:
        query_string: The text to search for.
        fuzzy_weight: Weight for the fuzzy (trigram) score in the combined
            score. Defaults to 0.5.
        semantic_weight: Weight for the semantic (vector) score in the
            combined score. Defaults to 0.5.
        unaccent: If True, normalize accents for fuzzy matching.
        semantic_threshold: Minimum cosine similarity for semantic results.
    """

    def __init__(
        self,
        query_string: str,
        fuzzy_weight: float = 0.5,
        semantic_weight: float = 0.5,
        unaccent: bool = False,
        semantic_threshold: float | None = None,
    ):
        self.query_string = query_string
        self.fuzzy_weight = fuzzy_weight
        self.semantic_weight = semantic_weight
        self.unaccent = unaccent
        self.semantic_threshold = semantic_threshold

    def __repr__(self):
        return (
            f"<Hybrid {repr(self.query_string)} "
            f"fuzzy_weight={self.fuzzy_weight} "
            f"semantic_weight={self.semantic_weight}>"
        )


class MatchAll(SearchQuery):
    def __repr__(self):
        return "<MatchAll>"


class Boost(SearchQuery):
    def __init__(self, subquery: SearchQuery, boost: float):
        self.subquery = subquery
        self.boost = boost

    def __repr__(self):
        return f"<Boost {repr(self.subquery)} boost={repr(self.boost)}>"


#
# Combinators
#


class And(SearchQuery):
    def __init__(self, subqueries):
        self.subqueries = subqueries

    def __repr__(self):
        return "<And {}>".format(
            " ".join(repr(subquery) for subquery in self.subqueries)
        )


class Or(SearchQuery):
    def __init__(self, subqueries):
        self.subqueries = subqueries

    def __repr__(self):
        return "<Or {}>".format(
            " ".join(repr(subquery) for subquery in self.subqueries)
        )


class Not(SearchQuery):
    def __init__(self, subquery: SearchQuery):
        self.subquery = subquery

    def __repr__(self):
        return f"<Not {repr(self.subquery)}>"


MATCH_ALL = MatchAll()
MATCH_NONE = Not(MATCH_ALL)
