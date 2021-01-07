from django import forms
from classifier import Form, ErrorMessage


class GuessForm(forms.Form):
    """URL情報フォームクラス

    """
    url = forms.URLField(
        label=Form.URL_LABEL_NAME,
        max_length=Form.URL_MAX_LENGTH,
        required=Form.URL_REQUIRED_FLAG,
        error_messages={'required': ErrorMessage.REQUIRED_MESSAGE,
                        'invalid': ErrorMessage.NO_EXIST_PAGE_MESSAGE},
        widget=forms.TextInput(
            attrs={'placeholder': Form.URL_PLACEHOLDER,
                   'class': Form.URL_CLASS_NAME}
        )
    )
