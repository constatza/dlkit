from dlkit.transforms import Transformation, Map


class Pipeline(Transformation):
    def __init__(self, *transformations):
        self.transformations = [
            transform
            for transform in transformations
            if transform is not None and isinstance(transform, Transformation)
        ]
        if len(self.transformations) == 0:
            # apply identity map if no transformations are provided
            self.transformations.append(Map(lambda x: x, lambda x: x))

    def add(self, transformation):
        if not isinstance(transformation, Transformation):
            raise TypeError("All transformations must inherit from BaseTransformation.")
        self.transformations.append(transformation)

    def apply(self, train, *args):
        train = self.fit_transform(train)
        args = [self.transform(arg) if arg else None for arg in args]
        return train, *args

    def fit(self, train):
        for transformation in self.transformations:
            transformation.fit(train)

    def fit_transform(self, train):
        for transformation in self.transformations:
            train = transformation.fit_transform(train)
        return train

    def transform(self, data):
        for transformation in self.transformations:
            data = transformation.transform(data)
        return data

    def inverse_transform(self, data):
        for transformation in self.transformations[::-1]:
            data = transformation.inverse_transform(data)
        return data

    def clean(self):
        for transformation in self.transformations:
            transformation.clean()
