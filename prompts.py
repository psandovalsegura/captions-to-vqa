user_prompts = {
    'v000': """Given a description of an image, write a question-answer pair. \
The question should not just ask for the subject of the image because the answer \
would be the caption itself. Focus on things like objects, attributes, colors, \
textures, patterns, shapes, sizes and scales, context, etc., but not all of these are applicable. \
Here is the image description: """,
    'v001': """Given a description of an image, write question-answer pairs. \
The question should not just ask for the subject of the image because the answer \
would be the caption itself. Focus on things like objects, attributes, colors, \
textures, patterns, shapes, sizes and scales, context, etc., but not all of these are applicable. \
Here is the image description: """,
    'v002': """Given a description of an image, write question-answer pairs. \
The question should not just ask for the subject of the image because the answer \
would be the caption itself. Focus on things like objects, attributes, colors, \
textures, patterns, shapes, sizes and scales, context, etc., but not all of these are applicable. \
If possible, prioritize questions starting with 'Which', 'Where', 'Who', 'Whose', 'Why', 'How', 'Are', and 'Is' instead of 'What'. \
Here is the image description: """,
    'v003': """Image Description: """,
}

system_prompts = {
    'v001': """Respond with only question-answer pairs in the following format, where the pairs are separated by newlines. You may provide one or many question-answer pairs, but do not deviate from this format:\n\
Q: <question>\n\
A: <answer>\n\
\n\
Q: <question>\n\
A: <answer>\n\
...\n\
""",
    'v002': """Respond with only question-answer pairs in the following format, where the pairs are separated by newlines. You may provide one or many question-answer pairs, but do not deviate from this format:\n\
Q: <question>\n\
A: <answer>\n\
\n\
Q: <question>\n\
A: <answer>\n\
...\n\
""",
    'v003': """You are a first-class language model transforming descriptions of images into questions with corresponding answers about those images. \
You are particularly good at generating challenging questions. \
For each image, you generate three question-answer pairs with their corresponding category. \
Do not add any other text besides the questions, answers, and categories.\n\
Possible categories for questions are: Number, Boolean, and Other. \
Questions for which the answer is numeric are categorized as "Number". Do not generate numeric questions if the description does not provide a clear number answer.\n\
Questions for which the answer is yes or no are categorized as "Boolean".\n\
All other questions are categorized as "Other".\n\
You do not need to state your assumptions.\n\
Here's an example:\n\
Image Description: "The image features a young woman facing the camera, with a serious expression. She has a banana positioned horizontally across her face, covering her mouth, making it look as if she has a banana smile. The backdrop appears to be a lined sheet of paper, with some numbers and possibly text written in the background, contributing to a creative or artistic visual style. The focus and lighting on the woman's face highlight her expression and the unusual use of the banana."\n\n\
Question: What is unusual about this photo?\n\
Answer: The woman uses a banana to cover her mouth.\n\
Category: Other\n\n\
Question: How many bananas are in the image?\n\
Answer: 1\n\
Category: Number\n\n\
Question: Is there text in the image?\n\
Answer: Yes\n\
Category: Boolean"""
}