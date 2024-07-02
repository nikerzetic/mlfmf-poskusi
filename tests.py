def test_generate_path_features_for_function():
    import entries_extractor as ee

    path = "D:\\Nik\\Projects\\mlfmf-poskusi\\stdlib\\code2vec\\train\\Agda.Builtin.Char_0003.dag"
    G, leaves, name = ee.extract_graph(path)
    features = ee.generate_path_features_for_function(G, leaves, 8, 2)
    print(features)


def test_format_as_label():
    """Test if output of entries_extractor.format_as_label matches the desired output:
    - -|h|_
    - is|idempotent|commutative|monoid
    - *|+|semiring
    - ¬|pn|<|1|+|v
    - ⟶|id
    - _|↠|∘|_
    - ∨|⊥|is|commutative|monoid
    - *|is|magma
    - *|is|commutative|monoid
    - *|monoid
    - x|∼|y|
    """
    import helpers
    import entries_extractor as ee

    test_cases = {
        "Data.Nat.Solver.+-*-Solver.-H_ 66": "-|h|_",
        "Data.Bool.Properties._.IsIdempotentCommutativeMonoid 162": "is|idempotent|commutative|monoid",
        "Data.Nat.Properties.*-+-semiring 6512": "*|+|semiring",
        "Data.Nat.Properties._._.¬Pn<1+v 6410": "¬|pn|<|1|+|v",
        "Function.Construct.Identity._.⟶-id 748": "⟶|id",
        "Function.Construct.Composition._↠-∘_ 2148": "_|↠|∘|_",
        "Algebra.Lattice.Properties.BooleanAlgebra.∨-⊥-isCommutativeMonoid 3176": "∨|⊥|is|commutative|monoid",
        "Algebra.Bundles.IdempotentSemiring._.*-isMagma 2576": "*|is|magma",
        "Algebra.Bundles.CommutativeRing._.*-isCommutativeMonoid 3694": "*|is|commutative|monoid",
        "Algebra.Bundles.Quasiring.*-monoid 2994": "*|monoid",
        "Data.Nat.Properties.∣m+n-m+o∣≡∣n-o| 6516": "",
        'x∼y"': "x|∼|y|",
    }
    results = {}
    for word in test_cases:
        label = ee.format_as_label(word)
        label = helpers.replace_unicode_with_latex(label)
        print(label)
        if label == test_cases[word]:
            continue
        results[word] = label


def test_replace_unicode_with_latex():
    import helpers
    import entries_extractor as ee

    test_cases = {
        "Data.Nat.Solver.+-*-Solver.-H_ 66": "-|h|_",
        "Data.Nat.Properties._._.¬Pn<1+v 6410": "\lnot|pn|<|1|+|v",
        'x∼y"': "x|\sim|y",
        "Data.Nat.Properties.∣m+n-m+o∣≡∣n-o| 6516": "",
    }

    results = {}
    for word in test_cases:
        new = ee.format_as_label(word)
        new = helpers.replace_unicode_with_latex(new)
        print(new)
        if new == test_cases[word]:
            continue
        results[word] = new


def test_create_dictionaries():
    import helpers

    raw2label, label2raw = helpers.create_dictionaries("stdlib")
    for name, label in raw2label.items():
        print(name, label, sep="\n", end="\n\n")


def test_dictionaries_and_embeddings():
    import helpers
    import json

    missings_in_embeddings = []
    missing_in_dictionaries = []

    raw2label, label2raw = helpers.create_dictionaries("stdlib")
    # f = open("data/raw/stdlib/dictionaries/raw2label.json", "r", encoding="utf-8")
    # raw2label: dict = json.load(f)
    # f.close()
    label2raw = {v: k for k,v in raw2label.items()}

    embeddings = helpers.read_embeddings("stdlib")

    for name, label in raw2label.items():
        if not label in embeddings:
            missings_in_embeddings.append(name)

    for label in embeddings:
        if not label in label2raw:
            missing_in_dictionaries.append(label)

    if len(missings_in_embeddings):
        print("Missing in embeddings:", len(missings_in_embeddings))
        print("\n\t".join(missings_in_embeddings))
    if len(missing_in_dictionaries):
        print("\nMissing in dictionaries:", len(missing_in_dictionaries))
        print("\n\t".join(missing_in_dictionaries))


if __name__ == "__main__":
    # test_generate_path_features_for_function()
    # test_replace_unicode_with_latex()
    # test_format_as_label()
    # test_create_dictionaries()
    test_dictionaries_and_embeddings()
