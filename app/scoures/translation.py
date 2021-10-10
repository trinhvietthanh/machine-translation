import six
import tensorflow as tf
import base64
import numpy as np


def to_example(dictionary):
    """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
    features = {}
    for (k, v) in six.iteritems(dictionary):
        if not v:
            raise ValueError("Empty generated field: %s" % str((k, v)))
        # Subtly in PY2 vs PY3, map is not scriptable in py3. As a result,
        # map objects will fail with TypeError, unless converted to a list.
        if six.PY3 and isinstance(v, map):
            v = list(v)
        if (isinstance(v[0], six.integer_types) or
                np.issubdtype(type(v[0]), np.integer)):
            features[k] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=v))
        elif isinstance(v[0], float):
            features[k] = tf.train.Feature(
                float_list=tf.train.FloatList(value=v))
        elif isinstance(v[0], six.string_types):
            if not six.PY2:  # Convert in python 3.
                v = [bytes(x, "utf-8") for x in v]
            features[k] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=v))
        elif isinstance(v[0], bytes):
            features[k] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=v))
        else:
            raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                             (k, str(v[0]), str(type(v[0]))))
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def translate(from_txt):
    # we need the argument direction here
    # so that translate(...) will not cache input
    # from envi to vien and vice versa.
    from_txt = from_txt.strip()
    input_ids = state.encoder.encode(from_txt) + [1]
    input_ids += [0] * (128 - len(input_ids))
    byte_string = to_example({
        'inputs': list(np.array(input_ids, dtype=np.int32))
    })
    content = base64.b64encode(byte_string).decode('utf-8')

