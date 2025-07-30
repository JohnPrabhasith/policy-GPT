initial_prompt = """
You are an expert at analyzing policy documents. Based on the following context, answer the question in detail.
If the answer is not available in the context, clearly state: "The answer is not available in the provided document." Do not invent information.

Context chunk:
{context}

Question:
{input}

Detailed Answer:
"""

refined_prompt = """
You are an expert at analyzing policy documents. Below is an existing detailed answer constructed so far, followed by new context.

Existing answer:
{existing_answer}

New context chunk:
{context}

Question:
{input}

Please update and refine the answer, adding any relevant details or conditions found in the new context.
If the new context doesn’t add any new information, return the existing answer unchanged. Do not invent information.

Refined Answer:
"""