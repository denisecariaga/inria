import numpy as np


def projection(vector, mu, dim1, dim2):
    vector_per_contact = np.split(vector, dim2 / dim1)
    projected = np.array([])

    for i in range(dim2 / dim1):
        mui = mu[i]
        x1 = vector_per_contact[i][0]
        normx2 = np.linalg.norm(vector_per_contact[i][1:])

        if normx2 <= (-mui) * x1:
            projected = np.concatenate((projected, np.zeros([dim1, ])))
        elif normx2 <= (1 / mui) * x1:
            projected = np.concatenate((projected, vector_per_contact[i]))
        else:
            x2 = vector_per_contact[i][1:]
            projected = np.concatenate((projected,
                                        (mui ** 2) / (1 + mui ** 2) * (x1 + (1 / mui) * normx2) * np.concatenate(
                                            (np.array([1]), (1 / mui) * x2 * (1 / normx2)))))

    return projected
