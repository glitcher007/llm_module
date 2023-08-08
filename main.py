import os
os.environ['OPENAI_API_KEY']='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'

from langchain.llms import OpenAI
llm=OpenAI(temperature=0.6)
name=llm("suggest hotel in rourkela")
#print(name)


from langchain.prompts import PromptTemplate
prompt_template_name=PromptTemplate(
    input_variables=['cuisine'],
    
    template="I want to open a restaurant for {cuisine} food.Suggest gancy name"
)

prompt_template_name.format(cuisine="punjabi")

from langchain.chains import LLMChain
chain=LLMChain(llm=llm,prompt=prompt_template_name)
chain.run("American")


llm = OpenAI(temperature=0.6)
prompt_template_name=PromptTemplate(
    input_variables=['cuisine'],
    
    template="I want to open a restaurant for {cuisine} food.Suggest fancy name."
)
name_chain = LLMChain(llm=llm,prompt=prompt_template_name)
prompt_template_items=PromptTemplate(
    input_variables=['restaurant_name'],
    
    template="Suggest food menu for{restaurant_name}.return it as coma seperated list."
)
food_items_chain=LLMChain(llm=llm,prompt=prompt_template_items)



from langchain.chains import SimpleSequentialChain
chain=SimpleSequentialChain(chains=[name_chain,food_items_chain])
response=chain.run("italian")
#print(response)

llm = OpenAI(temperature=0.6)
prompt_template_name=PromptTemplate(
    input_variables=['cuisine'],
    
    template="I want to open a restaurant for {cuisine} food.Suggest fancy name."
)
name_chain = LLMChain(llm=llm,prompt=prompt_template_name,output_key="restaurant_name")
prompt_template_items=PromptTemplate(
    input_variables=['restaurant_name'],
    
    template="Suggest food menu for{restaurant_name}.return it as coma seperated list."
)
food_items_chain=LLMChain(llm=llm,prompt=prompt_template_items,output_key="menu_items")


from langchain.chains import SimpleSequentialChain
chain =SimpleSequentialChain(chains=[name_chain,food_items_chain])
response=chain.run("Indian")
print(response)


llm = OpenAI(temperature=0.7)
prompt_template_name=PromptTemplate(
    input_variables=['cuisine'],
    
    template="I want to open a restaurant for {cuisine} food.Suggest fancy name."
)
name_chain = LLMChain(llm=llm,prompt=prompt_template_name,output_key="restaurant_name")
# input key remains the same the output key differ

llm=OpenAI(temperature=0.7)

prompt_template_items=PromptTemplate(
    input_variables=['restaurant_name'],
    template="Suggest some menu items for {restaurant_name},"
)
food_items_chain=LLMChain(llm=llm,prompt=prompt_template_items,output_key="menu_items")



from langchain.chains import SequentialChain
SequentialChain(
    chains=[name_chain,food_items_chain],
    input_variables=['cuisine','Arabic'],
    output_variables=['restaurant_name','menu_items']
)

chain({'cuisine': 'Arabic'})



