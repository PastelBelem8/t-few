from abc import ABC


class BaseTemplate(ABC):
    def __init__(self, config, answer_choices, *placeholders_cols):
        # We want to avoid the need for specifying on the config files
        # explicit column names. Instead, each reader will make the appropriate
        # assignment of columns to the templates. 
        # We assume the placeholder columns are passed in the filling-in same order.
        self.template2example = {f"s{i+1}": col for i, col in enumerate(placeholders_cols)}
        self.answer_choices = answer_choices

    def apply(self, example):

        # We get the placeholder values based on the map we created in config.
        placeholders_values = {placeholder: example[example_col] for placeholder, example_col in self.template2example.items()}
        input_str = self.template.format(**placeholders_values)

        if example["label"] == -1:
            target_str = "Unlabeled"
        else:
            target_str = self.answer_choices[example["label"]]

        return input_str, target_str

    def get_answer_choices_list(self, example):
        return self.answer_choices

    
class SemanticCovTemplate(BaseTemplate):
    def __init__(self, config, answer_choices, *placeholders_cols):
        super().__init__(config, answer_choices, *placeholders_cols)
        self.template = \
            "The sentence \"{s1}\" expresses several ideas. Are these ideas also expressed in the sentence below? Yes or No?\n\nSentence:\n{s2}"


class AdequacyTemplate(BaseTemplate):
    def __init__(self, config, answer_choices, *placeholders_cols):
        super().__init__(config, answer_choices, *placeholders_cols)
        self.template = "I want to know whether the two sentences mean the same thing.\n\n{s1}\n{s2}\nDo they?"
