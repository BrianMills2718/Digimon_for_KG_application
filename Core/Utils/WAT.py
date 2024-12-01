from dataclasses import dataclass, asdict

from pydantic import model_validator


@dataclass
class WATAnnotation:
    # An entity annotated by WAT

    def __init__(self, start, end, rho, explanation, spot, id, title):
        self.start = start  # char offset (included)

        self.end = end  # char offset (not included)

        self.rho = rho  # annotation accuracy

        self.explanation = explanation

        self.spot = spot  # annotated text

        # Wikipedia entity info
        self.wiki_id = id  # wiki_id
        self.wiki_title = title  # wiki_title

    @model_validator(mode="after")
    def __update_prior_prob(cls, data):
        if cls.explanation is not None:
            cls.prior_prob = cls.explanation['prior_explanation'][
                'entity_mention_probability']  # spot-entity probability
        return data

    @property
    def as_dict(self):
        return asdict(self)
