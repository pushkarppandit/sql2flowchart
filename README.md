# sql2flowchart
 Generates flowchart from sql select query

## Working
* Accepts a select query as a string input
* Converts query into a dict of 'simple' select queries, linked to each other through 'from' statements. A 'simple' select query is a select query without any subqueries inside 'from'.
* Converts parsed dict into a graph (nxgraph), each simple query is a node. Edges follow data flow.
* Graph is plotted using plotly's scatter plot. Hovering on a node shows additional information.

## Example
For an interactive notebook example look at Notebooks/Examples/example1.ipynb

```python
import sql2flowchart.SelectQuery as s2f
test_query = "<some select query>"
test_query_obj = s2f.SelectQuery(test_query) # create object
test_query_obj.parse_select_query() # parse query into dict
test_query_obj.plot_query() # create graph and plot fig
test_query_obj.fig.write_html('<path to output file>.html') # save interactive plot
```

## What is not supported
* Parsing select subqueries inside where, having and case statements. Only subqueries inside 'from' are parsed currently. Everything else is just represented as text in the hover text.
* Checking correctness of sql syntax: The input queries are expected to be syntactically correct. 

## Planned features for the future
* Table alias management: Table aliases referred to during column definitions can still be confusing. 