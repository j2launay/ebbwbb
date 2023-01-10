# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils import check_random_state

from .utils.gs_utils import get_distances
from . import growingspheres


class CounterfactualExplanation:
    """
    Class for defining a Counterfactual Explanation: this class will help point to specific counterfactual approaches
    """
    def __init__(self, obs_to_interprete, prediction_fn, method='GS', target_class=None, random_state=None, vectorizer=None,
                continuous_features=None, categorical_features=[], categorical_values=[]):
        """
        Init function
        method: algorithm to use
        random_state
        """
        if vectorizer is not None and type(obs_to_interprete) is str:
            # Vectorizing target instance
            self.obs_to_interprete = vectorizer.transform([obs_to_interprete])[0]
        else:
            self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        self.method = method
        self.target_class = target_class
        self.random_state = check_random_state(random_state)
        self.methods_ = {'GS': growingspheres.GrowingSpheres,
                         #'HCLS': lash.HCLS,
                         #'directed_gs': growingspheres.DirectedGrowingSpheres
                        }
        self.fitted = 0
        self.continuous_features = continuous_features
        self.categorical_features=categorical_features
        self.categorical_values = categorical_values
        
    def fit(self, caps=None, n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True, verbose=False, text=False, 
                        feature_variance=None, farthest_distance_training_dataset=None, probability_categorical_feature=None,
                        min_counterfactual_in_sphere=0, nlp=None):
        """
        find the counterfactual with the specified method
        """
        cf = self.methods_[self.method](self.obs_to_interprete,
                self.prediction_fn,
                self.target_class,
                caps,
                n_in_layer,
                first_radius,
                dicrease_radius,
                sparse,
                verbose,
                text=text,
                continuous_features=self.continuous_features, 
                categorical_features=self.categorical_features, 
                categorical_values=self.categorical_values,
                feature_variance=feature_variance,
                farthest_distance_training_dataset=farthest_distance_training_dataset,
                probability_categorical_feature=probability_categorical_feature,
                min_counterfactual_in_sphere=min_counterfactual_in_sphere,
                nlp=nlp)
        self.enemy, self.onevsrest, self.radius, self.iteration = cf.find_counterfactual()
        self.e_star = cf.e_star
        self.move = self.enemy - self.obs_to_interprete
        self.fitted = 1

    def distances(self, metrics=None):
        """
        scores de distances entre l'obs et le counterfactual
        """
        if self.fitted < 1:
            raise AttributeError('CounterfactualExplanation has to be fitted first!')
        return get_distances(self.obs_to_interprete, self.enemy)
    