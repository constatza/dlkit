type IntHyperparameter = int | dict[str, int] | dict[str, tuple[int, ...]]
type FloatHyperparameter = float | dict[str, float | int] | dict[str, tuple[float, ...]]
type StrHyperparameter = str | dict[str, str] | dict[str, tuple[str, ...]]

type Hyperparameter = IntHyperparameter | FloatHyperparameter | StrHyperparameter
