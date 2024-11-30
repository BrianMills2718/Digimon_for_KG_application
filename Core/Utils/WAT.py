from dataclasses import dataclass, asdict

@dataclass
class WATAnnotation:
    # An entity annotated by WAT

    def __init__(self, d):

        # char offset (included)
        self.start = d['start']
        # char offset (not included)
        self.end = d['end']

        # annotation accuracy
        self.rho = d['rho']
        # spot-entity probability
        self.prior_prob = d['explanation']['prior_explanation']['entity_mention_probability']

        # annotated text
        self.spot = d['spot']

        # Wikpedia entity info
        self.wiki_id = d['id']
        self.wiki_title = d['title']


    def as_dict(self):
        return asdict(self)