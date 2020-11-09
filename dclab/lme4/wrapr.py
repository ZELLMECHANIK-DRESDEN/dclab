"""R lme4 wrapper"""
import numbers
import warnings

import numpy as np

from .. import definitions as dfn
from ..rtdc_dataset.core import RTDCBase

from .rlibs import rpy2
from . import rsetup


class Lme4InstallWarning(UserWarning):
    pass


class Rlme4(object):
    def __init__(self, model="lmer", feature="deform"):
        """Perform an R-lme4 analysis with RT-DC data

        Parameters
        ----------
        model: str
            One of:

            - "lmer": linear mixed model using lme4's ``lmer``
            - "glmer+loglink": generalized linear mixed model using
              lme4's ``glmer`` with an additional a log-link function
              via the ``family=Gamma(link='log'))`` keyword.
        feature: str
            Dclab feature for which to compute the model

        References
        ----------
        .. [1] R package "lme4":
               Bates D, Maechler M, Bolker B and Walker S (2015). lme4:
               Linear mixed- effects models using Eigen and S4. R package
               version 1.1-9, https://CRAN.R-project.org/package=lme4.

        .. [2] R function "anova" from package "stats":
               Chambers, J. M. and Hastie, T. J. (1992) Statistical Models
               in S, Wadsworth & Brooks/Cole
        """
        #: modeling method to use (e.g. "lmer")
        self.model = None
        #: dclab feature for which to perform the analysis
        self.feature = None
        #: list of [RTDCBase, column, repetition, chip_region]
        self.data = []

        #: modeling function
        self.r_func_model = "feature ~ group + (1 + group | repetition)"
        self.r_func_nullmodel = "feature ~ (1 + group | repetition)"

        self.set_options(model=model, feature=feature)

        # Make sure that lme4 is available
        if not rsetup.has_lme4():
            warnings.warn("Installing lme4, this may take a while!",
                          Lme4InstallWarning)
            rsetup.install_lme4()

    def add_dataset(self, ds, group, repetition):
        assert group in ["treatment", "control"]
        assert isinstance(ds, RTDCBase)
        assert isinstance(repetition, numbers.Integral)
        self.data.append([ds, group, repetition,
                          ds.config["setup"]["chip region"]])

    def check_data(self):
        # Check that we have enough data
        if len(self.data) < 3:
            msg = "Linear Mixed Models require repeated measurements. " + \
                  "Please select more treatment repetitions."
            raise ValueError(msg)

    def fit(self, model=None, feature=None):
        """Perform G(LMM) fit

        Parameters
        ----------
        model: str (optional)
            One of:

            - "lmer": linear mixed model using lme4's ``lmer``
            - "glmer+loglink": generalized linear mixed model using
              lme4's ``glmer`` with an additional a log-link function
              via the ``family=Gamma(link='log'))`` keyword.
        feature: str (optional)
            Dclab feature for which to compute the model

        Returns
        -------
        results: dict
            The results of the entire fitting process:

            - "is differential": Boolean indicating whether or not
              the analysis was performed for the differential (bootstrapped
              and subtracted reservoir from channel data) feature
            - "feature": name of the feature used for the analysis
              `self.feature``
            - "fixed effects intercept": Mean of ``feature`` for all controls
            - "fixed effects treatment": The fixed effect size between the mean
              of the controls and the mean of the treatments
              relative to "fixed effects intercept"
            - "model": model name used for the analysis ``self.model``
            - "anova p-value": Anova likelyhood ratio test (significance)
            - "model summary": Summary of the model (exposed from R)
            - "model coefficients": Model coefficient table (exposed from R)
            - "r_err": errors and warnings from R
            - "r_out": standard output from R
        """

        self.set_options(model=model, feature=feature)
        self.check_data()

        # Assemble dataset
        if self.is_differential():
            # bootstrap and compute differential features using reservoir
            features, groups, repetitions = self.get_differential_dataset()
        else:
            # regular feature analysis
            features = []
            groups = []
            repetitions = []
            for dd in self.data:
                features.append(self.get_feature_data(dd[1], dd[2]))
                groups.append(dd[1])
                repetitions.append(dd[2])

        # Fire up R
        with rsetup.AutoRConsole() as ac:
            r = rpy2.robjects.r

            # Load lme4
            rpy2.robjects.packages.importr("lme4")

            # Concatenate huge arrays for R
            r_features = rpy2.robjects.FloatVector(np.concatenate(features))
            _groups = []
            _repets = []
            for ii in range(len(features)):
                _groups.append(np.repeat(groups[ii], len(features[ii])))
                _repets.append(np.repeat(repetitions[ii], len(features[ii])))
            r_groups = rpy2.robjects.StrVector(np.concatenate(_groups))
            r_repetitions = rpy2.robjects.IntVector(np.concatenate(_repets))

            # Register groups and repetitions
            rpy2.robjects.globalenv["feature"] = r_features
            rpy2.robjects.globalenv["group"] = r_groups
            rpy2.robjects.globalenv["repetition"] = r_repetitions

            # Create a dataframe which contains all the data
            r_data = r["data.frame"](r_features, r_groups, r_repetitions)

            # Random intercept and random slope model
            if self.model == 'glmer+loglink':
                r_model = r["glmer"](self.r_func_model, r_data,
                                     family=r["Gamma"](link='log'))
                r_nullmodel = r["glmer"](self.r_func_nullmodel, r_data,
                                         family=r["Gamma"](link='log'))
            else:  # lmer
                r_model = r["lmer"](self.r_func_model, r_data)
                r_nullmodel = r["lmer"](self.r_func_nullmodel, r_data)

            # Anova analysis
            anova = r["anova"](r_model, r_nullmodel, test="Chisq")
            pvalue = anova.rx["Pr(>Chisq)"][0][1]
            model_summary = r["summary"](r_model)
            coeff_summary = r["coef"](r_model)

            coeffs = r["data.frame"](r["coef"](model_summary))

            # TODO: find out what p.normal are for:
            # rpy2.robjects.globalenv["model"] = r_model
            # r("coefs <- data.frame(coef(summary(model)))")
            # r("coefs$p.normal=2*(1-pnorm(abs(coefs$t.value)))")

            fe_icept = coeffs[0][0]
            fe_treat = coeffs[1][0]
            if self.model == "glmer+loglink":
                # transform back from log
                fe_icept = np.exp(fe_icept)
                fe_treat = np.exp(fe_icept + fe_treat) - np.exp(fe_icept)

        ret_dict = {
            "anova p-value": pvalue,
            "differential feature": self.is_differential(),
            "feature": self.feature,
            "fixed effects intercept": fe_icept,
            "fixed effects treatment": fe_treat,  # aka "fixed effect"
            "model": self.model,
            "summary_model": model_summary,
            "summary_coefficients": coeff_summary,
            "r_err": ac.get_warnerrors(),
            "r_out": ac.get_prints(),
        }
        return ret_dict

    def get_differential_dataset(self):
        """

        The most famous use case is differential deformation. The idea
        is that you cannot tell what the difference in deformation
        from channel to reservoir is, because you never measure the
        same object in the reservoir and the channel. You usually just
        have two distributions. Comparing distributions is possible
        via bootstrapping. And then, instead of running the lme4
        analysis with the channel deformation data, it is run with
        the differential deformation (subtraction of the bootstrapped
        deformation distributions for channel and reservoir).
        """
        features = []
        groups = []
        repetitions = []
        # compute differential features
        for grp in sorted(set([dd[1] for dd in self.data])):
            for rep in sorted(set([dd[2] for dd in self.data])):
                feat_cha = self.get_feature_data(grp, rep, region="channel")
                feat_res = self.get_feature_data(grp, rep, region="reservoir")
                bs_cha, bs_res = bootstrapped_median_distributions(feat_cha,
                                                                   feat_res)
                # differential feature
                features.append(bs_cha - bs_res)
                groups.append(grp)
                repetitions.append(rep)
        return features, groups, repetitions

    def get_feature_data(self, group, repetition, region="channel"):
        assert group in ["control", "treatment"]
        assert isinstance(repetition, numbers.Integral)
        assert region in ["reservoir", "channel"]
        for dd in self.data:
            if dd[1] == group and dd[2] == repetition and dd[3] == region:
                ds = dd[0]
                break
        else:
            raise ValueError("Dataset for group '{}', repetition".format(group)
                             + " '{}', and region".format(repetition)
                             + " '{}' not found!".format(region))
        fdata = ds[self.feature]
        fdata_valid = fdata[~np.logical_or(np.isnan(fdata), np.isinf(fdata))]
        return fdata_valid

    def is_differential(self):
        """Return True if the differential feature is computed for analysis

        This effectively just checks the regions of the datasets
        and returns True if any one of the regions is "reservoir".

        See Also
        --------
        get_differential_features: for an explanation
        """
        for dd in self.data:
            if dd[3] == "reservoir":
                return True
        else:
            return False

    def set_options(self, model=None, feature=None):
        if model is not None:
            assert model in ["lmer", "glmer+loglink"]
            self.model = model
        if feature is not None:
            assert dfn.scalar_feature_exists(feature)
            self.feature = feature


def bootstrapped_median_distributions(a, b, bs_iter=1000, rs=117):
    """Compute a bootstrapped median distribution.

    Parameters
    ----------
    a, b: 1d ndarray of length N
        Input data
    bs_iter: int
        Number of bootstrapping iterations to perform
        (outtput size).
    rs: int
        Random state seed for random number generator

    Returns
    -------
    median_dist_a, median_dist_b: 1d arrays of length bs_iter
        Boostrap distribution of medians of `arr`
    """
    # Seed random numbers that are reproducible on different machines
    prng_object = np.random.RandomState(rs)
    # Initialize median arrays
    median_a = np.zeros(bs_iter)
    median_b = np.zeros(bs_iter)
    # If this loop is still too slow, we could get rid of it and
    # do everything with arrays. Depends on whether we will
    # eventually run into memory problems with array sizes
    # of y*bs_iter and yR*bs_iter.
    lena = len(a)
    lenb = len(b)
    for q in range(bs_iter):
        # Channel data:
        # Compute random indices and draw from y
        draw_a_idx = prng_object.randint(0, lena, lena)
        median_a[q] = np.median(a[draw_a_idx])
        draw_b_idx = prng_object.randint(0, lenb, lenb)
        median_b[q] = np.median(b[draw_b_idx])
    return median_a, median_b
