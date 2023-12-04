


class MitigationFramework:

    def __init__(self):
        return

    def run_explainer(self, explainer_method="integrated-gradients"):


        if explainer_method == "integrated-gradients":
            print("Integrated Gradients Explainer")
        elif explainer_method == "shap":
            print("SHAP Explainer")
        else:
            print("Unknown Explainer method")

        return

    def run_identifier(self, identifier_method="chatgpt"):

        if identifier_method == "chatgpt":
            print("ChatGPT Identifier")
        elif identifier_method == "mturk":
            print("MTurk Identifier")
        else:
            print("Unknown Identifier method")
        return

    def run_moderator(self):
        return