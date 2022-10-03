from abc import ABC


class BaseTemplate:
    def __init__(self, config, template: str, answer_choices=None, *placeholders_cols):
        self.is_regression_template = config.use_regress
        self.answer_choices = None if self.is_regression_template else answer_choices

        # We want to avoid the need for specifying on the config files
        # explicit column names. Instead, each reader will make the appropriate
        # assignment of columns to the templates. 
        # We assume the placeholder columns are passed in the filling-in same order.
        self.template2example = {f"s{i+1}": col for i, col in enumerate(placeholders_cols)}
        self.template = template

    def apply(self, example):
        # We get the placeholder values based on the map we created in config.
        placeholders_values = {placeholder: example[example_col] for placeholder, example_col in self.template2example.items()}
        input_str = self.template.format(**placeholders_values)

        if example["label"] == -1:
            target_str = "Unlabeled"
        else:
            label = example["label"]
            target_str = str(label) if self.is_regression_template else self.answer_choices[label]

        return input_str, target_str

    def get_answer_choices_list(self, example):
        if self.is_regression_template:
            # Regression is just syntatic sugar for token-wise decoding or
            # open-ended generation.
            return [str(example["label"])]
        else:
            return self.answer_choices


class CollectionTemplates(ABC):
    def __init__(self, templates, config, answer_choices, *placeholders_cols):
        self.all_template_names = []
        self.templates = {}

        for name, template in templates:
            self.all_template_names.append(name)
            self.templates[name] = BaseTemplate(config, template, answer_choices, *placeholders_cols)

    def __getitem__(self, key):
        return self.templates[key]

class SemanticCovTemplates(CollectionTemplates):
    def __init__(self, config, answer_choices, *placeholders_cols):
        if config.use_regress:
            templates = [
                ("basic", "Reference: \"{s1}\"\nIn a scale of 1 to 100 (with 1 being the lowest and 100 the highest) how confident are you that the following summary expresses all the ideas in the reference?\nSummary: {s2}"),
                # Inspired by T0 templates on word sense disambiguation
                ("prompt 1", "How much do you think \"{s2}\" expresses the same ideas as \"{s1}\"? Answer in a scale 1 to 100 (1 being the lowest and 100 the highest)."),
                ("prompt 2", "\"{s1}\"\n\"{s2}\"\nIn a scale from 1 to 100, how is summary 2 covering the ideas in the first summary?"),
                ("prompt 3", "Decide whether the second summary conveys the same ideas as the first summary:\n\"{s1}\"\n\"{s2}\"\nAnswer by yes or no."),
                ("prompt 4", "\"{s1}\"\n\"{s2}\"\nQuestion: Estimate the quality of the second paragraph as a paraphrase of the first paragraph. Use a scale 1 to 100."),
                ("prompt 5", "First read the summaries below.\n\nSummary 1: {s1}\nSummary 2: {s2}\nNow, would you say \"Summary 2\" expresses the same ideas as \"Summary 1\"? Use a 100-star rating (1 being the lowest and 100 the highest)."),
                ("prompt 6", "Homework\n\nSemantic coverage indicates whether the ideas in a Reference summary are mentioned in the hypothesis summary. Your homework is to determine whether following hypothesis summary below semantically covers the reference:\nReference: {s1}\nHypothesis: {s2}\nAnswer by yes or no."),
                # Inspired by App Reviews
                ("prompt 7", "Given this reference summary:\n\"{s1}\" would you say its ideas are clearly expressed in the summary below:\n\"{s2}\"?\nAnswer with a confidence between 1 and 100."),
                ("prompt 8", "Reference summary: {s2}\nWould you say that this summary describes the ideas in \"{s1}\"? Use a 1-100 scale."),          
            ]
        else:
            templates = [
                ("basic", "The sentence \"{s1}\" expresses several ideas. Are these ideas also expressed in the sentence below? Yes or No?\n\nSentence:\n{s2}"),
                # Inspired by T0 templates on word sense disambiguation
                ("prompt 1", "Do you think this summary \"{s2}\" expresses the same ideas as this one: \"{s1}\"?"),
                ("prompt 2", "\"{s1}\"\n\"{s2}\"\nDo you think summary 2 covers the same ideas as the first summary?"),
                ("prompt 3", "Decide whether the second summary conveys the same ideas as the first summary:\n\"{s1}\"\n\"{s2}\"\nAnswer by yes or no."),
                ("prompt 4", "\"{s1}\"\n\"{s2}\"\nQuestion: Is the second paragraph a good paraphrase of the first paragraph?"),
                ("prompt 5", "Summary 1: {s1}\nSummary 2: {s2}\nDetermine whether \"Summary 2\" expresses the same ideas as \"Summary 1\"."),
                ("prompt 6", "Homework\n\nSemantic coverage indicates whether the ideas in a Reference summary are mentioned in the hypothesis summary. Your homework is to determine whether following hypothesis summary below semantically covers the reference:\nReference: {s1}\nHypothesis: {s2}\nAnswer by yes or no."),
                # Inspired by App Reviews
                ("prompt 7", "Given this reference summary:\n\"{s1}\" would you say its ideas are clearly expressed in the summary below:\n\"{s2}\""),
                ("prompt 8", "Reference summary: {s2}\nWould you say that the summary above describes the ideas in \"{s1}\"?"),
            ]
        super().__init__(templates, config, answer_choices, *placeholders_cols)


class AdequacyTemplates(CollectionTemplates):
    def __init__(self, config, answer_choices, *placeholders_cols):
        if config.use_regress:
            templates = [
                ("basic", "Tell me how similar the following sentences are.\n\n{s1}\n{s2}\nUse a scale of 0 to 100 (where 0 is the lowest)?"),
                # Inspired by T0 templates on word sense disambiguation
                ("prompt 1", "Sentence 1: {s1}\nSentence 2: {s2}\nHow confident are you that these sentences have the exact same meaning?"),
                ("prompt 2", "Homework\n\n Find whether the following sentences express exactly the same:\nSentence: {s1}\nSentence: {s2}\nUse a 0-100 rating, where 0 means \"Not at all\" and 100 means \"Exactly the same\"?"),
                ("prompt 3", "A friend asked me whether the sentences \"{s1}\" and \"{s2}\" are paraphrases. What do you think?"),
                ("prompt 4", "{s1}\n\{s2}\n I know for a fact that these two sentences mean the same. Do you agree?"),
                ("prompt 5", "Given this reference translation: \"{s1}\", would you say \"{s2}\"\n means the same? Use a scale from 0-100."),
                ("prompt 6", "For a given reference translation, an hypothesis is deemed adequate if it represents the same ideas as the reference. Your homework is to find how adequate the following translation is:\n\nReference: {s1}\nHypothesis: {s2}\nAnswer with a value between 0 and 100."),
            ]
        else:
            templates = [
                ("basic", "I want to know whether the two sentences mean the same thing.\n\n{s1}\n{s2}\nDo they?"),
                # Inspired by T0 templates on word sense disambiguation
                ("prompt 1", "Sentence 1: {s1}\nSentence 2: {s2}\nDo you think these sentences have the exact same meaning?"),
                ("prompt 2", "Homework\n\n Find whether the following sentences express exactly the same:\nSentence: {s1}\nSentence: {s2}\nDo they?"),
                ("prompt 3", "A friend asked me whether the sentences \"{s1}\" and \"{s2}\" are paraphrases. What do you think?"),
                ("prompt 4", "{s1}\n\{s2}\n I know for a fact that these two sentences mean the same. Do you agree?"),
                ("prompt 5", "Given this reference translation: \"{s1}\", would you say \"{s2}\"\n means the same?"),
                ("prompt 6", "For a given reference translation, an hypothesis is deemed adequate if it represents the same ideas as the reference. Your homework is to find whether the following translation is adequate:\n\nReference: {s1}\nHypothesis: {s2}\nAnswer by yes or no."),
                # Inspired by App Reviews
            ]
        
        super().__init__(templates, config, answer_choices, *placeholders_cols)


class T5Templates(CollectionTemplates):
    def __init__(self, config, answer_choices, *placeholders_cols):
        templates = [
            ("stsb", "stsb sentence1: {s1} sentence2: {s2}."),         
            # ("realsumm", "semantics sentence1: {s1} sentence2: {s2}."),          
        ]
        super().__init__(templates, config, answer_choices, *placeholders_cols)