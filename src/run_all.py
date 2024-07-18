import numpy as np

from main import do_regression

if __name__ == "__main__":
    subject = "UTS02"
    n_delays = 5

    # envelope
    scores_envelope_train1 = do_regression(
        "envelope",
        n_stories=2,
        subject=subject,
        n_delays=n_delays,
        show_results=False,
    )
    scores_envelope_train3 = do_regression(
        "envelope",
        n_stories=4,
        subject=subject,
        n_delays=n_delays,
        show_results=False,
    )
    # scores_envelope_train5 = do_regression(
    #     "envelope",
    #     n_stories=6,
    #     subject=subject,
    #     n_delays=n_delays,
    #     show_results=False,
    # )
    # scores_envelope_train10 = do_regression(
    #     "envelope", n_stories=11, subject=subject, n_delays=n_delays, show_results=False,
    # )

    # TODO: shuffled

    # # embeddings
    # scores_embeddings_train1 = do_regression(
    #     "embeddings", n_stories=2, subject=subject, n_delays=n_delays, show_results=False,
    # )
    # scores_embeddings_train3 = do_regression(
    #     "embeddings", n_stories=4, subject=subject, n_delays=n_delays, show_results=False,
    # )
    # scores_embeddings_train5 = do_regression(
    #     "embeddings", n_stories=6, subject=subject, n_delays=n_delays, show_results=False,
    # )
    # scores_embeddings_train10 = do_regression(
    #     "embeddings", n_stories=11, subject=subject, n_delays=n_delays, show_results=False,
    # )

    np.save("data/scores_envelope_train1.npy", scores_envelope_train1)
    np.save("data/scores_envelope_train3.npy", scores_envelope_train3)
    # np.save("scores_envelope_train5", scores_envelope_train1)

    results_max = {
        "envelope": {
            "train1": scores_envelope_train1.max(),
            "train3": scores_envelope_train3.max(),
        }
    }

    # 1. save a json with restuls

    # 2. save a plot with results

    print("hi")
