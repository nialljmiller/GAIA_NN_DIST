import csv
import os  # Operating system related functionalities
from datetime import datetime  # Working with dates and times
import torch  # Main PyTorch library
from matplotlib import rcParams  # Customizing plot parameters
import numpy as np
import parallax_wjc as nnplx


def main(input_prefix: str = '', output_prefix: str = ''):
    """
    Main Driver function
    """

    epochs = 2048

    learn_rates = np.logspace(-4, 0, 3)
    learn_rates_gamma = np.linspace(0.0001, 1, 3)
    gaia_astrometric_weights = np.linspace(0.01, 1, 3)
    gaia_corr_weights = np.linspace(0.01, 1, 3)
    gaia_photometric_weights = np.linspace(0.01, 1, 3)
    vvv_astrometric_weights = np.linspace(0.01, 1, 3)
    vvv_corr_weights = np.linspace(0.01, 1, 3)
    vvv_photometric_weights = np.linspace(0.01, 1, 3)
    parallax_corr_over_errors_cut = np.linspace(0.1, 8, 3)
    ipd_frac_multi_peaks_cut = np.linspace(0.1, 10, 3)
    parallax_over_error_vvvs_cut = np.linspace(0.1, 10, 3)

    header = ['parallax_corr', 'phot_bp_rp_excess_factor_corr',
    'ra', 'dec', 'l', 'b', 'ecl_lon', 'ecl_lat',
    'parallax', 'pmra', 'pmdec',
    'dec_parallax_corr', 'dec_pmdec_corr', 'dec_pmra_corr',
    'parallax_pmdec_corr', 'parallax_pmra_corr',
    'pm', 'pmra_pmdec_corr', 'ra_dec_corr', 'radial_velocity',
    'ra_parallax_corr', 'ra_pmdec_corr', 'ra_pmra_corr',
    'ra_vvv', 'dec_vvv', 'l_vvv', 'b_vvv', 'parallax_vvv',
    'pmra_vvv', 'pmdec_vvv',
    'bp_g', 'bp_rp',
    'g_rp', 'grvs_mag',
    'J-K', 'H-K', 'Z-K', 'Y-K' 'epoch', 'learn_rate',
    'loss', 'delta_loss', 'counter', 'val_loss', 'r2_score', 'mse', 'mae']

    with open(output_prefix + 'output_card.csv', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(header)

    for learn_rate in learn_rates:
        for learn_rate_gamma in learn_rates_gamma:
            for parallax_corr_over_error_cut in parallax_corr_over_errors_cut:
                for ipd_frac_multi_peak_cut in ipd_frac_multi_peaks_cut:
                    for parallax_over_error_vvv_cut in parallax_over_error_vvvs_cut:
                        for astrometric_weight in gaia_astrometric_weights:
                            for corr_weight in gaia_corr_weights:
                                for photometric_weight in gaia_photometric_weights:

                                    gaia_astrometric_weight = astrometric_weight
                                    gaia_corr_weight = corr_weight
                                    gaia_photometric_weight = photometric_weight
                                    vvv_astrometric_weight = astrometric_weight
                                    vvv_corr_weight = corr_weight
                                    vvv_photometric_weight = photometric_weight

                                    config_list = [learn_rate, learn_rate_gamma, gaia_astrometric_weight,
                                                   gaia_corr_weight, gaia_photometric_weight,
                                                   vvv_astrometric_weight,
                                                   vvv_corr_weight, vvv_photometric_weight,
                                                   parallax_corr_over_error_cut,
                                                   ipd_frac_multi_peak_cut, parallax_over_error_vvv_cut]
                                
                                    config_string = ''.join([str(round(num * 100)) for num in config_list])

                                    x_filter = ['parallax_corr', 'phot_bp_rp_excess_factor_corr',
                                                'ra', 'dec', 'l', 'b', 'ecl_lon', 'ecl_lat',
                                                'parallax', 'pmra', 'pmdec',
                                                'dec_parallax_corr', 'dec_pmdec_corr', 'dec_pmra_corr',
                                                'parallax_pmdec_corr', 'parallax_pmra_corr',
                                                'pm', 'pmra_pmdec_corr', 'ra_dec_corr', 'radial_velocity',
                                                'ra_parallax_corr', 'ra_pmdec_corr', 'ra_pmra_corr',
                                                'ra_vvv', 'dec_vvv', 'l_vvv', 'b_vvv', 'parallax_vvv',
                                                'pmra_vvv', 'pmdec_vvv',
                                                'bp_g', 'bp_rp',
                                                'g_rp', 'grvs_mag',
                                                'J-K', 'H-K', 'Z-K', 'Y-K']

                                    x_importance_raw = [gaia_astrometric_weight, gaia_corr_weight,
                                                    gaia_astrometric_weight, gaia_astrometric_weight,
                                                    gaia_astrometric_weight, gaia_astrometric_weight,
                                                    gaia_astrometric_weight, gaia_astrometric_weight,
                                                    gaia_astrometric_weight, gaia_astrometric_weight,
                                                    gaia_astrometric_weight,
                                                    gaia_corr_weight, gaia_corr_weight, gaia_corr_weight,
                                                    gaia_corr_weight, gaia_corr_weight,
                                                    gaia_astrometric_weight, gaia_corr_weight,
                                                    gaia_corr_weight, gaia_astrometric_weight,
                                                    gaia_corr_weight, gaia_corr_weight, gaia_corr_weight,
                                                    vvv_astrometric_weight, vvv_astrometric_weight,
                                                    vvv_astrometric_weight, vvv_astrometric_weight,
                                                    vvv_astrometric_weight,
                                                    vvv_astrometric_weight, vvv_astrometric_weight,
                                                    gaia_photometric_weight, gaia_photometric_weight,
                                                    gaia_photometric_weight, gaia_photometric_weight,
                                                    vvv_photometric_weight, vvv_photometric_weight,
                                                    vvv_photometric_weight, vvv_photometric_weight]
                                    
                                    x_importance = []
                                    jitter = 0.1
                                    for ximp in x_importance_raw:
                                        new_weight = -1
                                        while new_weight < 0:
                                            new_weight = ximp * jitter * np.random.uniform(0, 1)
                                        x_importance.append(new_weight)

                                    goodata_filters = [
                                        ('parallax_corr_over_error',
                                         lambda x: abs(x) > parallax_corr_over_error_cut),
                                        ('ipd_frac_multi_peak', lambda x: abs(x) < ipd_frac_multi_peak_cut),
                                        ('parallax_over_error_vvv',
                                         lambda x: abs(x) > parallax_over_error_vvv_cut)
                                    ]

                                    data_fp = input_prefix + 'VVV_GAIA_STANDARDS.csv'
                                    data_verif_fp = data_fp  # input_prefix + 'GAIA_VARS.csv'

                                    assert len(x_importance) == len(x_filter)

                                    y_filter = ['parallax_corr']

                                    output_dir = output_prefix + str(epochs) + '_' + str(
                                        int(1 / learn_rate)) + '_' + str(
                                        int(10000 * learn_rate_gamma)) + '_' + config_string + '/'

                                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

                                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                    print(device)

                                    train_dl, test_dl, vector_len = nnplx.prepare_data(data_fp, x_filter,
                                                                                       x_importance,
                                                                                       y_filter,
                                                                                       goodata_filters)
                                    print('Data prepared')

                                    model = nnplx.MLP(vector_len)
                                    print('Model compiled')

                                    if True:  # not os.path.exists(output_dir + 'model.pt'):
                                        if not os.path.exists(output_dir):
                                            os.mkdir(output_dir)
                                        train_results = nnplx.train_model(train_dl, test_dl, model, epochs,
                                                                          learn_rate, output_dir,
                                                                          learn_rate_gamma)
                                        torch.save(model.state_dict(), output_dir + 'model.pt')

                                        # Append new_data to existing_data
                                        results_line = x_importance + train_results

                                        # Write the combined data (existing_data + new_data) to the CSV file
                                        with open(output_prefix + 'output_card.csv', 'a',
                                                  newline='') as file:
                                            csv_writer = csv.writer(file)
                                            csv_writer.writerows(results_line)

                                        print('Model trained')
                                    else: 
                                        print('Loading model from here : ', output_dir + 'model.pt')
                                        model.load_state_dict(torch.load(output_dir + 'model.pt'))

                                    print('Predicting distances')
                                    nnplx.predict_dist(model, data_fp, data_verif_fp, output_dir, x_filter,
                                                       x_importance, y_filter)
                                    return


# YOU CAN USE CEPHIED AND AGB PERIOD LUMINMOSITY RELATIONS TO VERIFY
if __name__ == '__main__':
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    dpi = 200  # 200-300 as per guidelines
    maxpix = 670  # max pixels of plot
    width = maxpix / dpi  # max allowed with
    rcParams.update({'axes.labelsize': 'large', 'axes.titlesize': 'large',  # the size of labels and title
                     'xtick.labelsize': 'large', 'ytick.labelsize': 'large',  # the size of the axes ticks
                     'legend.fontsize': 'medium', 'legend.frameon': False,  # legend font size, no frame
                     'legend.facecolor': 'none', 'legend.handletextpad': 0.1,
                     # legend no background colour, separation from label to point
                     'font.serif': ['Computer Modern', 'Helvetica', 'Arial',  # default fonts to try and use
                                    'Tahoma', 'Lucida Grande', 'DejaVu Sans'],
                     'font.family': 'serif',  # use serif fonts
                     'mathtext.fontset': 'cm', 'mathtext.default': 'regular',  # if in math mode, use these
                     'figure.figsize': [width, 0.7 * width], 'figure.dpi': dpi,
                     # the figure size in inches and dots per inch
                     'lines.linewidth': .75,  # width of plotted lines
                     'xtick.top': True, 'ytick.right': True,  # ticks on right and top of plot
                     'xtick.minor.visible': True, 'ytick.minor.visible': True,  # show minor ticks
                     'text.usetex': True})  # process text with LaTeX instead of matplotlib math mode
    main('OUTPUT/', 'GAIA_NN_DIST_DATA/')
