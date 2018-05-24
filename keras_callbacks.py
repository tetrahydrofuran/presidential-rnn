import keras
import pandas as pd
import lib


class SentenceGenerator(keras.callbacks.Callback):

    def __init__(self, output_path=None, verbose=0):
        super(SentenceGenerator, self).__init__()
        self.output_path = output_path
        self.verbose = verbose
        self.sentences = pd.DataFrame(columns=['epoch', 'seed', 'diversity', 'generated_post'])

    def on_epoch_end(self, epoch, logs={}):

        # Reference variables
        sentence_agg = list()

        seed_chars = '<Hillary Clinton>'[:lib.window_len]

        for diversity in [0.2, 0.5, 1.0, 1.2]:

            generated = ''
            sentence = seed_chars
            generated += sentence

            # Generate next characters, using a rolling window
            for next_char_index in range(lib.predict_len):
                x_pred, text_y = lib.gen_x_y(sentence, false_y=True)

                preds = self.model.predict(x_pred, verbose=0)[-1]

                next_index = lib.sample(preds, diversity)
                next_char = lib.index_to_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

            local_dict = dict()
            local_dict['epoch'] = epoch
            local_dict['seed'] = seed_chars
            local_dict['diversity'] = diversity
            local_dict['generated_post'] = generated
            sentence_agg.append(local_dict)

            if self.verbose >= 1:
                print('Diversity: {}, generated post: {}'.format(local_dict['diversity'], local_dict['generated_post']))

        epoch_sentences = pd.DataFrame(sentence_agg)

        self.sentences = pd.concat(objs=[self.sentences, epoch_sentences])
        if self.output_path is not None:
            self.sentences.to_csv(path_or_buf=self.output_path, index=False)