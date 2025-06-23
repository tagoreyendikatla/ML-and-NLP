1. CountVectorizer (Bag-of-Words)
    Implementation: Simple word counting

    Scoring: Raw frequency counts

    Characteristics:

    No normalization

    No weighting scheme

    Common words get high scores

2. Manual TF-IDF
    Implementation: Custom Python implementation

    Scoring: TF × IDF

    TF = term frequency in document

    IDF = log(total docs/docs containing term)

    Characteristics:

    Basic IDF calculation

    No smoothing

    No normalization

 3. Scikit-learn's TfidfVectorizer
    Implementation: Optimized sklearn implementation

    Scoring: Enhanced TF-IDF

    Smoothed IDF (log[(1 + N)/(1 + df)] + 1)

    L2 normalization by default

    Characteristics:

    More robust to edge cases

    Normalized outputs

    Better for machine learning   