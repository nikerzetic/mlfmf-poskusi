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
        'x∼y"': "x|∼|y|",
    }
    results = {}
    for word in test_cases:
        label = ee.format_as_label(word)
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
    }

    results = {}
    for word in test_cases:
        new = ee.format_as_label(word)
        new = helpers.replace_unicode_with_latex(new)
        print(new)
        if new == test_cases[word]:
            continue
        results[word] = new


if __name__ == "__main__":
    # test_generate_path_features_for_function()
    test_replace_unicode_with_latex()
