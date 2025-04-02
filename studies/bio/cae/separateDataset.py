import random
import numpy as np

checkPsolutionsLoad = np.load(
    r"M:\constantinos\data\bio\5-equations\solutions\solutions.npy", allow_pickle=True
).T

random.seed(1234)
readFile = r"M:\constantinos\data\bio\5-equations\solutions\porousSolutions.npy"
Solutions = np.load(readFile, allow_pickle=True).T
Solutions = np.transpose(Solutions, (2, 0, 1))
# Solutions = Solutions[:, :, :-8]

# Read Udofs.txt file correctly
Udofs = np.loadtxt(r"M:\constantinos\data\bio\5-equations\dofs\Udofs.txt")

Udofs = Udofs.astype(int)
Solutions = np.transpose(Solutions, (0, 2, 1))
Solutions = Solutions[:, Udofs, :]

saveFile = r"M:\constantinos\data\bio\5-equations\solutions\Usolutions.npy"
np.save(saveFile, Solutions)

a = 1
