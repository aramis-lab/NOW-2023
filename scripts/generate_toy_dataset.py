import argparse
import pandas as pd
from sklearn.datasets import make_classification


def _dynamic_padding(number: int, width: int) -> str:
    return "sub-{number:0{width}d}".format(width=width, number=number)


def main(n_samples: int, output_filename: str) -> None:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=2,
        flip_y=0.2,
        weights=[0.5,0.5],
        random_state=17,
    )
    df = pd.DataFrame(X, columns=["HC_left_volume", "HC_right_volume"])
    df["group"] = y
    df["group"] = df["group"].apply(lambda x: "AD" if x==1 else "CN")
    n_digits = len(str(n_samples))
    df["subject_id"] = [_dynamic_padding(x, n_digits) for x in range(1, len(df) + 1)]
    df.set_index('subject_id', inplace=True)
    df.to_csv(output_filename, sep="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_samples", help="The number of samples to generate.", type=int)
    parser.add_argument("output", help="Output file name.", type=str)
    args = parser.parse_args()
    main(n_samples=args.n_samples, output_filename=args.output)

