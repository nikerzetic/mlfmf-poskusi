# mlfmf-poskusi

Najprej je potrebno pobrati datoteke iz [repozitorija](https://zenodo.org/records/10041075). Razširimo jih v glavno mapo repozitorija:

> mlfmf-poskusi/
>   ...
>   mathlib/
>   stdlib/
>   TypeTopology/
>   unimath/

Potem poženemo `load_library` v `helpers.py` na željeni knjižnici (hardcode-ano v zadnjem delu datoteke). Z `probibalistic_copy_entries_into_train_val_test_directories` razdelimo entries na tri datoteke, kot to zahteva `code2vec`. Z ukazom `reindex_library_asts` popravimo indekse AST-jev.

Iz mape `entries` dobimo primerno datoteko za code2vec učenje z `bash` ukazom:

> sh mlfmf-poskusi/code2vec-master/preprocess.sh

v kateri je potrebno popraviti določene spremenljivke.

Z ukazom:

> sh code2vec-master/train.sh

poženemo učenje (vendar tega še nisem usposobil).

