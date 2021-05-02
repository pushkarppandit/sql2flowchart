# sql2flowchart
 Generates flowchart from sql select query

## Working
* Accepts a select query as a string input
* Converts query into a dict of 'simple' select queries, linked to each other through 'from' statements. 
    * A 'simple' select query is a select query without any subqueries inside 'from'.
    * Subqueries inside 'from' statements are replaced by a randomly generated name and the query is pulled out into a separate simple query
* Converts parsed dict into a graph (nxgraph), each simple query is a node. Edges follow data flow.
* Graph is plotted using plotly's scatter plot. Hovering on a node shows additional information.

## Example
For an interactive notebook example look at Notebooks/Examples/example1.ipynb

```python
import sql2flowchart.SelectQuery as s2f
test_query = "<some select query>"
test_query_obj = s2f.SelectQuery(test_query) # create object
test_query_obj.parse_select_query() # parse query into dict
test_parsed_dict = test_query_obj.all_queries # parsed dictionary
test_query_obj.plot_query() # create graph and plot fig
test_query_obj.fig.write_html('<path to output file>.html') # save interactive plot
```

For the input query
```sql
with 
combined as (
    select aa.id,
    a_sum,
    b_sum
    from (
        select a_id as id,
        sum(a) as a_sum
        from `schema.aaa`
        group by 1
    ) aa
    left join 
    (
        select b_id as id,
        sum(b) as b_sum
        from `schema.bbb`
        group by 1
    ) bb
    on aa.id = bb.id
),
c as (
    select id,
    a_sum,
    b_sum
    from `schema.old_agg`
    )
select * from combined
union all
select * from c
```
you get this parsed query
```json
{
  "combined": {
    "select": {
      "id": "aa.id",
      "a_sum": "a_sum",
      "b_sum": "b_sum"
    },
    "from": {
      "input_tables": {
        "aa": "XaL8w01b",
        "bb": "uL0mnQCV"
      },
      "joins": {
        "bb": {
          "type": "left",
          "on": "aa.id = bb.id"
        }
      },
      "combine_type": "join"
    },
    "where": "",
    "group": [],
    "having": ""
  },
  "c": {
    "select": {
      "id": "id",
      "a_sum": "a_sum",
      "b_sum": "b_sum"
    },
    "from": {
      "input_tables": {
        "`schema.old_agg`": "`schema.old_agg`"
      }
    },
    "where": "",
    "group": [],
    "having": ""
  },
  "5AidgeWq": {
    "select": {
      "*": "*"
    },
    "from": {
      "input_tables": {
        "combined": "combined"
      }
    },
    "where": "",
    "group": [],
    "having": ""
  },
  "u6jQKm4F": {
    "select": {
      "*": "*"
    },
    "from": {
      "input_tables": {
        "c": "c"
      }
    },
    "where": "",
    "group": [],
    "having": ""
  },
  "out": {
    "select": {
      "*": "*"
    },
    "from": {
      "input_tables": {
        "5AidgeWq": "5AidgeWq",
        "u6jQKm4F": "u6jQKm4F"
      },
      "combine_type": "union"
    },
    "where": "",
    "group": [],
    "having": ""
  },
  "XaL8w01b": {
    "select": {
      "id": "a_id",
      "a_sum": "sum(a)"
    },
    "from": {
      "input_tables": {
        "`schema.aaa`": "`schema.aaa`"
      }
    },
    "where": "",
    "group": [
      "id"
    ],
    "having": ""
  },
  "uL0mnQCV": {
    "select": {
      "id": "b_id",
      "b_sum": "sum(b)"
    },
    "from": {
      "input_tables": {
        "`schema.bbb`": "`schema.bbb`"
      }
    },
    "where": "",
    "group": [
      "id"
    ],
    "having": ""
  }
}
```

and this plot (static here, for the dynamic version please look at the notebook example3 / html output):
![example flow](https://github.com/pushkarppandit/sql2flowchart/blob/main/readme_meta/example_flow_static.png?raw=true)

## What is not supported
* Parsing select subqueries inside where, having and case statements. Only subqueries inside 'from' are parsed currently. Everything else is just represented as text in the hover text.
* Checking correctness of sql syntax: The input queries are expected to be syntactically correct. 

## Planned features for the future
* Table alias management: Table aliases referred to during column definitions can still be confusing. 