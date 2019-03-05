from __future__ import division, print_function, absolute_import
from PIL import Image
import numpy as np
from music21 import converter,instrument, duration, chord, stream # or import *
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from os import walk

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import datetime
from random import randint
import os
import time
from keras import backend as K
from random import choice


class Main:

    def __init__(self):
        self.notes = None
        self.t = None

    def get_notes(self):
        allnotesl = []
        notes = []

        midis = []
        for (dirpath, dirnames, filenames) in walk("midis"):
            midis.extend(filenames)
            break

        for midi in midis:
            if midi:
                s = converter.parse('midis/{}'.format(midi))
                print("Preparing {}".format(midi))

                i = 0
                for el in s.recurse():
                    dur = el.duration.quarterLength
                    note = str(el).split(" ")[-1]
                    note = note.replace(">", "")

                    if dur > 10:
                        el.duration.quarterLength = 5

                    if dur <= 0.1:
                        el.duration.quarterLength = 0.1

                    if 'Instrument' in el.classes:  # or 'Piano'
                        el.activeSite.replace(el, instrument.Harp())

                    notelength = [note, el.duration.quarterLength]
                    allnotesl.append(notelength)

                # s.write('midi', 'example.mid')

                for notel in allnotesl:
                    if isinstance(notel[0], str):
                        notes.append(str(notel[0]))

        t = Tokenizer(lower=False, filters="")
        # fit the tokenizer on the documents
        oof = t.fit_on_texts(notes)

        # summarize what was learned
        print("Keras Length, ", t.document_count)
        print("Allnotes length: ", len(allnotesl))
        self.notes = allnotesl
        return t


    def make_data(self, t):
        w, h = 128, 128
        data = np.zeros((h, w, 3), dtype=np.uint8)
        i = 0
        frame_number = 1
        truers = False
        for note in self.notes:
            for x in range(w):
                        key = None
                        if truers:
                            truers = False
                            break
                        if x == i:
                            if i % 127 == 0 and i != 0:
                                img = Image.fromarray(data, "RGB")
                                img.save("train/frame{}.png".format(frame_number))
                                #img.show()
                                frame_number += 1
                                data = np.zeros((h, w, 3), dtype=np.uint8)
                                i = 0
                                break
                            for y in range(h):
                                if not key:
                                    for number, notekey in t.index_word.items():
                                        if note[0] == notekey:
                                            key = number
                                            print("Found notekey " + notekey)

                                if y == key:
                                    data[key, x] = [255, 255, 255]
                                    print("Created ({},{})".format(key, x))
                                    truers = True
                                    i += 1
                                    break



if __name__ == "__main__":
    engine = Main()
    note_data = engine.get_notes()
    engine.make_data(note_data)








