from . import utils
from . import anchor_base
from . import anchor_explanation
from . import co_occurrence_matrix as co_occ
import numpy as np
import json
import os
import string
import sys
from io import open
os.environ['SPACY_WARNING_IGNORE'] = 'W008'

# Python3 hack
try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    def unicode(s):
        return s


def id_generator(size=15):
    """Helper function to generate random div ids. This is useful for embedding
    HTML into ipython notebooks."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(np.random.choice(chars, size, replace=True))

class AnchorText(object):
    """bla"""
    def __init__(self, nlp, class_names, use_unk_distribution=True, mask_string='UNK'):
        """
        Args:
            nlp: spacy object
            class_names: list of strings
            use_unk_distribution: if True, the perturbation distribution
                will just replace words randomly with mask_string.
                If False, words will be replaced by similar words using word
                embeddings
            mask_string: String used to mask tokens if use_unk_distribution is True.
        """
        self.nlp = nlp
        self.class_names = class_names
        self.neighbors = utils.Neighbors(self.nlp)
        self.use_unk_distribution = use_unk_distribution
        self.mask_string = mask_string
        # Generates and store the co-ocurence matrix  
        #co_occurence = co_occ.generate_co_occurence_matrix()
        # Stores the n best words from the co-occurence matrix
        #self.n_best_co_occurrence = co_occ.generate_n_best_co_occurrence(co_occurence)

    def get_sample_fn(self, text, classifier_fn, use_proba=False, pertinents_negatifs=False, pertinents_negatifs_replace=False):
        true_label = classifier_fn([text])
        if pertinents_negatifs:
            # Generates a new sentence based on the target added with the false pertinents words 
            # i.e: target sentence = "This is a good book" sentence_false_pertinent = "This is a very good scientific book" 
            # with 'very' and 'scientific' as false pertinents words
            sentence_false_pertinents  = utils.generate_false_pertinent(
                    text, [], 1, self.neighbors, self.n_best_co_occurrence, use_proba=use_proba, generate_sentence=True)
            text_pertinent = sentence_false_pertinents
            processed = self.nlp(unicode(text_pertinent))
            words = [x.text for x in processed]
            # Positions in the sentence of the beginning of each word
            positions = [x.idx for x in processed]
        elif pertinents_negatifs_replace:
            # Generates a new sentence based on the target added with the false pertinents words 
            # i.e: target sentence = "This is a good book" sentence_false_pertinent = "This is a very good scientific book" 
            # with 'very' and 'scientific' as false pertinents words
            sentence_false_pertinents_replace  = utils.generate_false_pertinents_replace(
                    text, [], 1, self.neighbors, self.n_best_co_occurrence, use_proba=use_proba, generate_sentence=True)
            text_pertinent = sentence_false_pertinents_replace
            processed = self.nlp(unicode(text_pertinent))
            words = [x.text for x in processed]
            # Positions in the sentence of the beginning of each word
            positions = [x.idx for x in processed]
        else:
            processed = self.nlp(unicode(text))
            words = [x.text for x in processed]
            # Positions in the sentence of the beginning of each word
            positions = [x.idx for x in processed]

        def sample_fn(present, num_samples, compute_labels=True, pyTorch=False, pertinents_negatifs=False, pertinents_negatifs_replace=False):
            # Generates 'num_samples' random sentences with presence or absence of certain words of the target sentence
            # i.e: A matrix of 1 or 0 with 1 meaning presence of the words and 0 absence
            if self.use_unk_distribution:
                # Replace absent words with a mask
                # data corresponds to the matrix of 1 and 0 while raw represents the matrix of sentences
                data = np.ones((num_samples, len(words)))
                raw = np.zeros((num_samples, len(words)), '|S80')
                raw[:] = words
                for i, t in enumerate(words):
                    if i in present:
                        continue
                    n_changed = np.random.binomial(num_samples, .5)
                    changed = np.random.choice(num_samples, n_changed,
                                               replace=False)
                    raw[changed, i] = self.mask_string
                    data[changed, i] = 0
                if (sys.version_info > (3, 0)):
                    raw_data = [' '.join([y.decode() for y in x]) for x in raw]
                else:
                    raw_data = [' '.join(x) for x in raw]
            else:
                if pertinents_negatifs:
                    # Modify the target sentence with false pertinent words randomly present or missing
                    data, raw_data, sentence_false_pertinents  = utils.generate_false_pertinent(
                        text, present, num_samples, self.neighbors, self.n_best_co_occurrence, use_proba=use_proba)
                elif pertinents_negatifs_replace:
                    data, raw_data, sentence_false_pertinents  = utils.generate_false_pertinents_replace(
                        text, present, num_samples, self.neighbors, self.n_best_co_occurrence, use_proba=use_proba)
                else:
                    # Perturb the target sentence to replace missing words with word with similar meaning
                    raw_data, data = utils.perturb_sentence(
                        text, present, num_samples, self.neighbors, top_n=100,
                        use_proba=use_proba)
            labels = []
            if true_label == 1:
                true_labels = np.ones(num_samples)
            else:
                true_labels = np.zeros(num_samples)
            compute_label = classifier_fn(raw_data)

            if compute_labels and pyTorch:
                # Compute labels with model using pyTorch
                labels = (classifier_fn(raw_data) == true_label).int()
            elif compute_labels: 
                labels = compute_label == true_labels
                #labels = int((classifier_fn(raw_data) == true_label))
            labels = np.array(labels)
            raw_data = np.array(raw_data).reshape(-1, 1)
            if pertinents_negatifs or pertinents_negatifs_replace and not self.use_unk_distribution:
                return raw_data, data, labels, sentence_false_pertinents
            return raw_data, data, labels
        return words, positions, true_label, sample_fn

    def explain_instance(self, text, classifier_fn, threshold=0.95,
                          delta=0.1, tau=0.15, batch_size=100, use_proba=False,
                          beam_size=4, pertinents_negatifs=False, pertinents_negatifs_replace=False,
                          **kwargs):
        # words corresponds to the different words from the target sentence, positions to their positions in the sentence
        # true_labels to the label predicted by the black box and sample_fn is the function generating the matrix of perturbed sentence
        words, positions, true_label, sample_fn = self.get_sample_fn(
            text, classifier_fn, use_proba=use_proba, pertinents_negatifs=pertinents_negatifs, 
            pertinents_negatifs_replace=pertinents_negatifs_replace)
        # Main function that find an anchor based on the matrix of perturbed sentence
        exp = anchor_base.AnchorBaseBeam.anchor_beam(
            sample_fn, delta=delta, epsilon=tau, batch_size=batch_size,
            desired_confidence=threshold, stop_on_first=True, pertinents_negatifs=pertinents_negatifs, 
            pertinents_negatifs_replace=pertinents_negatifs_replace, **kwargs) 
        if pertinents_negatifs:
            exp['names'] = ['not ' + words[x] for x in exp['feature']]
        elif pertinents_negatifs_replace:
            exp['names'] = ['not ' + words[x] for x in exp['feature']]
        else: 
            exp['names'] = [words[x] for x in exp['feature']]
        exp['positions'] = [positions[x] for x in exp['feature']]
        exp['instance'] = text
        exp['prediction'] = true_label
        explanation = anchor_explanation.AnchorExplanation('text', exp,
                                                           self.as_html)
        return explanation

    def as_html(self, exp):
        predict_proba = np.zeros(len(self.class_names))
        exp['prediction'] = int(exp['prediction'])
        predict_proba[exp['prediction']] = 1
        predict_proba = list(predict_proba)

        def jsonize(x):
            return json.dumps(x)
        this_dir, _ = os.path.split(__file__)
        bundle = open(os.path.join(this_dir, 'bundle.js'), encoding='utf8').read()
        random_id = 'top_div' + id_generator()

        example_obj = []

        def process_examples(examples, idx):
            idxs = exp['feature'][:idx + 1]
            out_dict = {}
            new_names = {'covered_true': 'coveredTrue', 'covered_false': 'coveredFalse', 'covered': 'covered'}
            for name, new in new_names.items():
                ex = [x[0] for x in examples[name]]
                out = []
                for e in ex:
                    processed = self.nlp(unicode(str(e)))
                    raw_indexes = [(processed[i].text, processed[i].idx, exp['prediction']) for i in idxs]
                    out.append({'text': e, 'rawIndexes': raw_indexes})
                out_dict[new] = out
            return out_dict

        example_obj = []
        for i, examples in enumerate(exp['examples']):
            example_obj.append(process_examples(examples, i))

        explanation = {'names': exp['names'],
                       'certainties': exp['precision'] if len(exp['precision']) else [exp['all_precision']],
                       'supports': exp['coverage'],
                       'allPrecision': exp['all_precision'],
                       'examples': example_obj}
        processed = self.nlp(unicode(exp['instance']))
        raw_indexes = [(processed[i].text, processed[i].idx, exp['prediction'])
                       for i in exp['feature']]
        raw_data = {'text': exp['instance'], 'rawIndexes': raw_indexes}
        jsonize(raw_indexes)

        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        out += u'''
        <div id="{random_id}" />
        <script>
            div = d3.select("#{random_id}");
            lime.RenderExplanationFrame(div,{label_names}, {predict_proba},
            {true_class}, {explanation}, {raw_data}, "text", "anchor");
        </script>'''.format(random_id=random_id,
                            label_names=jsonize(self.class_names),
                            predict_proba=jsonize(list(predict_proba)),
                            true_class=jsonize(False),
                            explanation=jsonize(explanation),
                            raw_data=jsonize(raw_data))
        out += u'</body></html>'
        return out

    def show_in_notebook(self, exp, true_class=False, predict_proba_fn=None):
        """Bla"""
        out = self.as_html(exp, true_class, predict_proba_fn)
        from IPython.core.display import display, HTML
        display(HTML(out))
