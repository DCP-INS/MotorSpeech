#!/usr/bin/env python3

from statsmodels.formula.api import ols
import statsmodels.api as sm
import rpy2.robjects.packages as rpackages
from rpy2.robjects.conversion import localconverter, rpy2py
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
import pandas as pd
import numpy as np
from os.path import join as pjoin
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


d = {}

# Initialize required R packages
base = importr('base')
utils = rpackages.importr('utils')
packnames = ('lme4', 'lmerTest', 'emmeans', 'mvtnorm',)
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

# helper functions
def jn(mlist, char=None):
    """Summary

    Parameters
    ----------
    mlist : list
       List of strings to concatenate.
    join : str, optional
       Character for concatenation.

    Returns
    -------
    Concatenated string
    """
    if not char:
        char = '_'
    string = char.join(mlist)
    return string

def flatten(nested):
    """flatten nested list.

    Parameters
    ----------
    nested : list
        list containing multiple lists
    n : None, optional
        dimension of the arrays in the lists

    Returns
    -------
    list
    """
    flatl = [item for sublist in nested for item in sublist]
    empty_l = [np.asarray(flatl).shape[i]
               for i in range(np.asarray(flatl).ndim)]

    tup = tuple(empty_l)
    flat_list = np.array(flatl).reshape(tup)

    return flat_list

color = {
    'timings': {
        'inlab': [
            # #EDDAB8
            np.array((237, 218, 184)).reshape(1, -1),
            # #BF5C5D
            np.array((191, 92, 93)).reshape(1, -1),
            # #46607A
            np.array((70, 96, 122)).reshape(1, -1),
            # #52474D
            np.array((82, 71, 77)).reshape(1, -1),
            # #4D395B
            np.array((77, 57, 91)).reshape(1, -1)
        ],
        'online': [
            # --honeydew: #f1faeeff;
            np.array((241, 250, 238)).reshape(1, -1),
            # --powder-blue: #a8dadcff;
            # np.array((168, 218, 220)).reshape(1, -1),
            np.array((135, 170, 187)).reshape(1, -1),
            # --celadon-blue: #457b9dff;
            np.array((69, 123, 157)).reshape(1, -1),
            # --prussian-blue: #1d3557ff;
            np.array((29, 53, 87)).reshape(1, -1)
        ]
    },
    'semantics': {
        'online': [
            # --honeydew: #e0f2e9ff;
            np.array((236, 108, 4)).reshape(1, -1),
            # --wintergreen-dream: #679289ff;
            np.array((241, 152, 79)).reshape(1, -1),
            # np.array((118, 156, 148)).reshape(1, -1),
            # np.array((103, 146, 137)).reshape(1, -1),
            # --myrtle-green: #1d7874ff;
            np.array((0, 121, 105)).reshape(1, -1),
            # --dark-jungle-green: #071e22ff;
            np.array((102, 174, 165)).reshape(1, -1)
            # np.array((7, 30, 34)).reshape(1, -1)
        ]
    }

class MotorSpeech():
    """
    A class to analyze motor speech data using both Python and R for statistical modeling.

    Attributes:
        main (str): The main experimental setup, either 'online' or 'inlab'.
        task (str): The specific task being analyzed.
        dv (str): The dependent variable for the analysis.
        save_keys (tuple): A tuple containing main, task, and dv for saving results.
    """

    def __init__(self, main, task, dv):
        self.main = main
        self.task = task
        self.dv = dv
        self.save_keys = (main, task, dv)

    ##########################################################################
    # Prepare data frames
    ##########################################################################

    def prepare_data(self):
        """
        Prepares and filters the data for analysis.

        Returns:
            pd.DataFrame: The prepared and filtered dataframe.
        """
        with pd.HDFStore(pjoin('data', jn([self.main, self.task, 'data.h5'])), mode='r') as store:
            df = store.select('df')

        # remove latency under 200 ms and missed trials
        df = df.loc[(df['rt'] > .2) & (df['misses'].isna())]

        # select only good subjects
        df = self.split_sujs(df)

        # add columns inv_efficiency and normalize for plotting
        df = self.add_inv(df)

        # select only correct for rt and inverse efficiency
        if self.dv != 'accuracy':
            df = df.loc[(df['accuracy'] == 1)]

        return df

    def split_sujs(self, df, lcutoff=None):
        """
        Splits subjects into good and bad based on accuracy scores and filters the data accordingly.

        Args:
            df (pd.DataFrame): The dataframe to be split.
            lcutoff (int, optional): The cutoff score for good subjects. Defaults to None.

        Returns:
            pd.DataFrame: The filtered dataframe with only good subjects.
        """
        # hardcode one-sided binomial cutoffs
        if not lcutoff:
            lcutoff = 38 if self.main == 'inlab' else 63
        good_s = []
        bad_s = []
        for subject in np.unique(df['subject'].values):
            bad = False
            for condition in np.unique(df['condition'].values):
                df_t = df.loc[(df['subject'] == subject) &
                              (df['condition'] == condition)]
                score = np.nanmean(df_t['accuracy']) * 100
                if np.isnan(score):
                    continue
                elif score < lcutoff:
                    bad = True
            if np.isnan(score):
                continue
            elif bad:
                bad_s.append(subject)
            else:
                good_s.append(subject)

        return df.loc[(df['subject'].isin(good_s))]

    def add_inv(self, df):
        """
        Add columns inverse efficiency and prepare normalized columns for later plotting.

        Returns:
            pd.DataFrame: Dataframe with added columns.
        """
        # overall mean rt and mean accuracy
        all_rt = np.nanmean(df['rt'].values)
        all_ac = np.nanmean(df['accuracy'].values)

        # loop over conditions
        for condition, subject in product(
            np.unique(
                df['condition']), np.unique(
                df['subject'])):
            df_out = df.loc[(df['subject'] == subject)
                            & (df['accuracy'].notna())]
            df_j = df
            if len(df_out) == 0:
                continue

            # subject mean rt and mean accuracy
            subject_rt = np.nanmean(df_out['rt'].values)
            subject_ac = np.nanmean(df_out['accuracy'].values)

            # select only condition
            con_df = df_out.loc[(df_out['condition'] == condition)]
            df_j = con_df.index
            cor_df = con_df.loc[(con_df['accuracy'] == 1)]
            df_i = con_df.index[(con_df['accuracy'] == 1)]
            df.loc[df_i, 'inv_eff'] = cor_df['logrt'].values / \
                np.nanmean(con_df['accuracy'].values)

            # normalize for plotting
            df.loc[df_i, 'norm_rt'] = cor_df['rt'].values - \
                subject_rt + all_rt
            df.loc[df_j, 'norm_accuracy'] = con_df['accuracy'].values - \
                subject_ac + all_ac

        return df

    ##########################################################################
    # Statistics in python and r
    ##########################################################################

    def lmm_python(self, df):
        """
        Performs linear mixed modeling using Python's statsmodels.

        Args:
            df (pd.DataFrame): The dataframe for analysis.

        Returns:
            None
        """
        add_blocks = False
        reference = 'control' if self.main == 'inlab' else 'double' if self.task == 'timings' else 'con_yes'
        df["group"] = 1
        vcf = {"subject": "0 + C(subject)", "stimulus": "0 + C(stimulus)"}
        vcf_b = {
            "subject": "0 + C(subject)",
            "stimulus": "0 + C(stimulus)",
            "blocknr": "0 + C(blocknr)"}

        base = sm.MixedLM.from_formula(
            "%s ~ 1" %
            (self.dv),
            groups="group",
            vc_formula=vcf,
            data=df).fit(
            reml=False)

        block = sm.MixedLM.from_formula(
            "%s ~ 1" %
            (self.dv),
            groups="group",
            vc_formula=vcf_b,
            data=df).fit(
            reml=False)

        if base.aic > block.aic:
            add_blocks = True

        if self.main == 'inlab':
            result = sm.MixedLM.from_formula(
                "%s ~ C(condition, Treatment(reference='%s'))" %
                (self.dv, reference), groups="group", vc_formula=vcf_b, data=df).fit(
                reml=False)

        elif self.main == 'online':
            if self.task == "timings":
                c1 = 'beeps'
                c2 = 'motor'
            else:
                c1 = 'congruency'
                c2 = 'articulation'
            result = sm.MixedLM.from_formula(
                "%s ~ %s" % (self.dv, '1 + %s * %s' % (c1, c2)),
                groups="group", vc_formula=vcf_b, data=df).fit(reml=False)

        results_df = pd.DataFrame({
            'Parameter': result.params.index,
            'Estimate': result.params.values,
            'P-value': result.pvalues
        })

        results_df.to_hdf(pjoin('data', 'motorspeech_results.h5'),
                          key='/'.join(self.save_keys) + '/pylmm',
                          mode='a')

        return add_blocks

    def lmm_r(self, df, add_blocks=True):
        """
        Performs linear mixed modeling using R's lme4 package.

        Args:
            df (pd.DataFrame): The dataframe for analysis.
            add_blocks (bool, optional): Whether to include block as a random effect. Defaults to True.

        Returns:
            None
        """
        predictors = "condition + (1|subject) + (1|stimulus)"
        if add_blocks:
            predictors += " + (1|blocknr)"
        lmm = ro.r('''
                   library(lme4)
                   library(lmerTest)
                   library(emmeans)

                   emm_options(lmerTest.limit = 7000)
                   emm_options(pbkrtest.limit = 7000)

                   m1 = lmer(%s ~ %s, data = df)
                   summary(m1)
                   ''' % (self.dv, predictors))

        with localconverter(ro.default_converter + pandas2ri.converter):
            todf = rpy2py(lmm)
        todf.to_hdf(pjoin('data', 'motorspeech_results.h5'),
                    key='/'.join(self.save_keys) + '/rlmm',
                    mode='a')

    def emmeans_r(self, df, add_blocks=True):
        """
        Performs estimated marginal means analysis using R's emmeans package.

        Args:
            df (pd.DataFrame): The dataframe for analysis.
            add_blocks (bool, optional): Whether to include block as a random effect. Defaults to True.

        Returns:
            None
        """
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df)

        ro.r('library(lme4)')
        ro.r('library(lmerTest)')
        ro.r('library(emmeans)')

        ro.r('emm_options(lmerTest.limit = 7000)')
        ro.r('emm_options(pbkrtest.limit = 7000)')

        ro.globalenv['r_df'] = r_df
        ro.globalenv['dv'] = self.dv
        predictors = "condition + (1|subject) + (1|stimulus)"
        if add_blocks:
            predictors += " + (1|blocknr)"
        ro.globalenv['predictors'] = predictors

        # test binomial for accuracy
        # m1 = glmer(as.formula(paste(dv, "~", predictors)), data = r_df,
        # family = binomial)
        ro.r('''
            m1 = lmer(as.formula(paste(dv, "~", predictors)), data = r_df)
            m1_emm = emmeans(m1, specs = "condition")
            test = pairs(m1_emm)
            summary(test)
            ''')

        pemmeans = ro.r('summary(test)')

        with localconverter(ro.default_converter + pandas2ri.converter):
            todf = rpy2py(pemmeans)
        todf.to_hdf(pjoin('data', 'motorspeech_results.h5'),
                    key='/'.join(self.save_keys) + '/emmeans',
                    mode='a')

    ##########################################################################
    # Make figures
    ##########################################################################

    def ms_figure(self, df, plot):
        # order conditions
        if self.main == 'online':
            if self.task == 'timings':
                t_conds = ['audiomotor', 'motor', 'audio', 'control'][::-1]
            else:
                t_conds = ['S+A+', 'S+A-', 'S-A+', 'S-A-']
        else:
            t_conds = ['control', 'phrasal', 'lexical', 'syllabic']

        # figure settings
        fsize_ax = 21
        colors = rigid.plot_colors()
        if self.dv == 'accuracy':
            fig_ylabel = 'Proportion correct (%)'
        elif self.dv == 'logrt':
            fig_ylabel = 'Correct RT (sec)'
        elif self.dv == 'inv_eff':
            fig_ylabel = 'IES (sec)'

        # prepare some empty lists
        labels_l = []
        bars = []

        # loop over conditions
        for i, condition in enumerate(t_conds):
            c_bars = []

            # loop over individual subjects for right tail cutoffs
            for s, subject in enumerate(np.unique(df['subject'])):

                # select only subject and task
                df_out = df.loc[(df['subject'] == subject) &
                                (df['condition'] == condition)]
                if len(df_out) == 0:
                    continue

                # fill lists to concatenate subjects
                norm_rts = df_out['norm_rt']
                norm_acs = df_out['norm_accuracy']
                if self.dv == 'accuracy':
                    c_bars.append(np.nanmean(norm_acs) * 100)
                elif self.dv == 'logrt':
                    c_bars.append(np.nanmean(norm_rts))
                elif self.dv == 'inv_eff':
                    c_bars.append(np.nanmean(norm_rts) / np.nanmean(norm_acs))

            # fill mean for each subject
            labels_l.append([condition] * len(c_bars))
            bars.append(np.array(c_bars))

        if plot == 'ms':
            # start plotting
            fig, ax = plt.subplots(1, figsize=(9, 9))
            n_df = pd.DataFrame(
                {'labels': flatten(labels_l), 'data': flatten(np.array(bars))})

            sns.swarmplot(x="labels",
                          y="data",
                          size=5,
                          color='k',
                          data=n_df,
                          zorder=3,
                          ax=ax)
            sns.boxplot(x="labels",
                        y="data",
                        notch=True,
                        palette=[colors[self.task][self.main]
                                 [i] / 255 for i in range(len(t_conds))],
                        width=.5,
                        data=n_df,
                        zorder=2,
                        ax=ax)
            ax.tick_params(axis='y', labelsize=fsize_ax)
            ax.set_ylabel(fig_ylabel, fontsize=fsize_ax + 8)
            ax.set_xlabel('')
            ax.set_xticks(range(len(t_conds)))
            ax.set_xticklabels([jn(x.split('_'), char='\n')
                                for x in t_conds], fontsize=fsize_ax + 4)

        else:
            result = pd.read_hdf(pjoin('data', 'motorspeech_results.h5'),
                                 key='/'.join(self.save_keys) + '/emmeans')
            fig, axes = plt.subplots(
                1, len(
                    result['contrast']), figsize=(
                    17, 5), sharey=False)
            # loop over all pairwised contrasts
            for j, pair in enumerate(result['contrast']):
                ax = axes[j]
                # read emmeans dataframe
                tdf = result.loc[(result['contrast'] == pair)]
                # get names specific two condition
                c1, c2 = pair.split(' - ')
                # get mean data of subjects for each condition
                c1d, c2d = bars[t_conds.index(
                    c1[1:-1])], bars[t_conds.index(c2[1:-1])]

                # create new dataframe for plotting
                n_df = pd.DataFrame(
                    {'labels': [c1[1:-1]] * len(c1d) + [c2[1:-1]] * len(c2d),
                     'data': np.concatenate([c1d, c2d]),
                     'subject': np.concatenate([range(len(c1d)), range(len(c2d))])})

                # make the boxplot plus dots regardless significance
                sns.swarmplot(x="labels",
                              y="data",
                              size=3,
                              color='k',
                              linewidth=1,
                              data=n_df,
                              zorder=3,
                              ax=ax)
                sns.boxplot(x="labels", y="data", notch=True, linewidth=1,
                            palette=[colors[self.task][self.main][i] / 255
                                     for i in (
                                         t_conds.index(c1[1: -1]),
                                         t_conds.index(c2[1: -1]))],
                            width=.5, data=n_df, zorder=3, ax=ax)
                if tdf['p.value'].values < .05:
                    signs = c1d - c2d
                    x = np.nanmedian(signs)
                    same_sign_i = [
                        ii for ii, num in enumerate(signs) if (
                            num < 0 if x < 0 else num > 0)]
                    other_sign_i = [
                        ii for ii, num in enumerate(signs) if (
                            num > 0 if x < 0 else num < 0)]
                    for k, s in enumerate((other_sign_i, same_sign_i)):
                        ndf = n_df.loc[(n_df['subject'].isin(s))]
                        sns.lineplot(data=ndf,
                                     x="labels",
                                     y="data",
                                     estimator=None,
                                     units="subject",
                                     color="red" if k == 0 else "green",
                                     linewidth=1,
                                     ax=ax)

                else:
                    sns.lineplot(data=n_df,
                                 x="labels",
                                 y="data",
                                 estimator=None,
                                 units="subject",
                                 color="grey",
                                 linewidth=1,
                                 ax=ax)
                ax.set_xlabel('')
                if j == 0:
                    ax.set_ylabel(fig_ylabel, fontsize=fsize_ax + 4)
                else:
                    ax.set_ylabel('')
        plt.tight_layout()

def main(args):
    """
    Main function to run the MotorSpeech analysis.

    Args:
        args (dict): Dictionary containing 'main', 'task', and 'dv' for the analysis.

    Returns:
        None
    """
    ms = MotorSpeech(main=args['main'], task=args['task'], dv=args['dv'])

    df = ms.prepare_data()

    add_blocks = ms.lmm_python(df)
    ms.emmeans_r(df)  # , add_blocks=add_blocks)

    for plot in ('ms', 'paired'):
        ms.ms_figure(df, plot=plot)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--main',
        type=str,
        help='The main experimental setup, either "online" or "inlab".')
    parser.add_argument(
        '-t',
        '--task',
        type=str,
        help='The specific task being analyzed.')
    parser.add_argument(
        '-d',
        '--dv',
        type=str,
        help='The dependent variable for the analysis.')
    arguments = vars(parser.parse_args())

    main(arguments)
