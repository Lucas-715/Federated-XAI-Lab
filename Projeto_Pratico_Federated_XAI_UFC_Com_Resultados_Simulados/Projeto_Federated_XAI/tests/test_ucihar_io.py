import numpy as np
from flxai.data import uci_har_dataset

def test_uci_official_file_layout(tmp_path):
    root=tmp_path/'UCI HAR Dataset'
    for split,subjects in [('train',list(range(1,11))),('test',[11,12,13])]:
        folder=root/split;folder.mkdir(parents=True)
        rng=np.random.default_rng(7)
        X=rng.normal(size=(len(subjects)*3,12))
        Y=np.tile(np.array([1,2,3]),len(subjects))
        G=np.repeat(subjects,3)
        np.savetxt(folder/f'X_{split}.txt',X,fmt='%.6f')
        np.savetxt(folder/f'y_{split}.txt',Y,fmt='%d')
        np.savetxt(folder/f'subject_{split}.txt',G,fmt='%d')
    (root/'features.txt').write_text(''.join(f'{i} sensor_{i}\n' for i in range(1,13)))
    data=uci_har_dataset(root,seed=7)
    assert data.X_train.shape[1]==12
    assert len(data.y_test)==9
    assert all(0<=x<6 for x in data.y_train)
    assert len(data.group_train)>0
    assert len(data.features)==12
