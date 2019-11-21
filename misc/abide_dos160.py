import numpy as np
from nilearn.datasets import fetch_abide_pcp
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib import cm


# rois_aal (116), rois_cc200 (200), rois_cc400 (392), rois_dosenbach160 (161), rois_ez (116), rois_ho (111), rois_tt (97)

abide = fetch_abide_pcp(pipeline='cpac',
                        band_pass_filtering=True,
                        global_signal_regression=True,
                        derivatives=['rois_dosenbach160'],
                        quality_checked=True
                        )

dir = '/content/gdrive/My Drive/Colab Notebooks/Project'

dos = abide.rois_dosenbach160

cor = ConnectivityMeasure(kind='correlation').fit_transform(dos)
in_cor = ConnectivityMeasure(kind='partial correlation').fit_transform(dos)

data_dir = os.path.join(dir, 'dos160')
cov_dir = os.path.join(data_dir, 'cov')
incov_dir = os.path.join(data_dir, 'incov')

os.makedirs(data_dir, exist_ok = True)
os.makedirs(cov_dir, exist_ok = True)
os.makedirs(incov_dir, exist_ok = True)

for idx in range(len(dos)):

    file_name = '/' + str(abide.phenotypic['SUB_ID'][idx]) + '.txt'
    np.savetxt(cov_dir + file_name, cor[idx])
    np.savetxt(incov_dir + file_name, in_cor[idx])

print("Finished correlation computation.")

plotting.plot_matrix(cor[0], figure=(9, 7), title='Covariance') # vmax=1, vmin=-1,
plotting.plot_matrix(in_cor[0], figure=(9, 7), title='Inverse covariance') # np.log(in_cor[0]+2 +1e-16)
plotting.show()
