from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField
from wtforms.validators import DataRequired


class MainForm(FlaskForm):
    input_field1 = SelectField('SelectField1',
                               choices=[('p', 'project'), ('w', 'ward'), ('s', 'street'), ('d', 'district')])
    input_item1 = StringField('TextInput1')

    input_field2 = SelectField('SelectField2',
                               choices=[('p', 'project'), ('w', 'ward'), ('s', 'street'), ('d', 'district')])
    input_item2 = StringField('TextInput2')

    input_field3 = SelectField('SelectField3',
                               choices=[('p', 'project'), ('w', 'ward'), ('s', 'street'), ('d', 'district')])
    input_item3 = StringField('TextInput3')

    input_field4 = SelectField('SelectField4',
                               choices=[('p', 'project'), ('w', 'ward'), ('s', 'street'), ('d', 'district')])
    input_item4 = StringField('TextInput4')

    input_field5 = SelectField('SelectField5',
                               choices=[('p', 'project'), ('w', 'ward'), ('s', 'street'), ('d', 'district')])
    input_item5 = StringField('TextInput5')

    submit = SubmitField('Submit')
