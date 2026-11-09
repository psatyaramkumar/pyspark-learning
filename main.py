from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import time

def setup_spark_session():
    """Initialize Spark session with optimal configuration for teaching"""
    return SparkSession.builder \
        .appName("SparkSQL_Teaching_Demo") \
        .master("local[2]") \
        .config("spark.sql.autoBroadcastJoinThreshold", "10MB") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.sql.adaptive.enabled", "false") \
        .getOrCreate()

def create_sample_data(spark):
    """Create comprehensive sample datasets for all demonstrations"""
    
    # Create employees DataFrame
    employees_data = [
        (1, "Alice", "Sales", 5000, ["Java", "Python"]),
        (2, "Bob", "Marketing", 4500, ["SQL", "Excel"]),
        (3, "Charlie", "Sales", 5500, ["Python", "Spark"]),
        (4, "Diana", "Engineering", 7000, ["Java", "Spark", "Kafka"]),
        (5, "Eve", "Engineering", 6500, ["Python", "Airflow"]),
        (6, "Frank", None, 4000, ["Excel"]),
        (7, "Grace", "Marketing", 4800, ["SQL", "Tableau"]),
        (8, "Henry", "Sales", 5200, ["Java", "SQL"]),
        (9, "Ivy", "Engineering", None, ["Spark", "Scala"]),
        (10, "Jack", "Sales", 4700, ["Python", "Excel"])
    ]
    
    employees_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dept", StringType(), True),
        StructField("salary", IntegerType(), True),
        StructField("skills", ArrayType(StringType()), True)
    ])
    
    employees_df = spark.createDataFrame(employees_data, schema=employees_schema)
    
    # Create departments DataFrame
    departments_data = [
        ("Sales", "New York", 1000000),
        ("Marketing", "Chicago", 800000),
        ("Engineering", "San Francisco", 1500000),
        ("HR", "Boston", 500000)
    ]
    
    departments_df = spark.createDataFrame(departments_data, 
                                           ["dept_name", "location", "budget"])
    
    # Create sales DataFrame for UNION demo
    sales_q1 = [("NY", "Alice", 1000), ("CA", "Bob", 2000)]
    sales_q2 = [("NY", "Alice", 1500), ("CA", "Charlie", 2500), ("NY", "Alice", 500)]
    
    sales_q1_df = spark.createDataFrame(sales_q1, ["state", "rep", "amount"])
    sales_q2_df = spark.createDataFrame(sales_q2, ["state", "rep", "amount"])
    
    # Create arrays DataFrame for LATERAL VIEW demo
    arrays_data = [
        (1, ["apple", "banana", "cherry"]),
        (2, ["dog", "cat"]),
        (3, ["single_item"]),
        (4, ["first", "second", "third", "fourth"])
    ]
    
    arrays_df = spark.createDataFrame(arrays_data, ["id", "items"])
    
    return {
        "employees": employees_df,
        "departments": departments_df,
        "sales_q1": sales_q1_df,
        "sales_q2": sales_q2_df,
        "arrays": arrays_df
    }

def register_initial_views(spark, dataframes):
    """Register all DataFrames as temporary views before starting"""
    print("\nRegistering DataFrames as temporary views...")
    
    # Clear existing temporary views
    for table in spark.catalog.listTables():
        if table.isTemporary:
            spark.catalog.dropTempView(table.name)
    
    dataframes["employees"].createOrReplaceTempView("employees")
    dataframes["departments"].createOrReplaceTempView("departments")
    dataframes["sales_q1"].createOrReplaceTempView("sales_q1")
    dataframes["sales_q2"].createOrReplaceTempView("sales_q2")
    dataframes["arrays"].createOrReplaceTempView("arrays")
    
    print("Registered views:", [table.name for table in spark.catalog.listTables()])
    print()

def part1_sql_execution(spark, dataframes):
    """Part 1: Introduction to SQL Execution in Spark"""
    print("\n" + "="*80)
    print("PART 1: SQL Execution in Spark")
    print("="*80)
    
    employees_df = dataframes["employees"]
    
    print("\n1.1 Direct SQL Execution with spark.sql():")
    print("-" * 40)
    print("Query: SELECT dept, AVG(salary) as avg_salary FROM employees GROUP BY dept")
    sql_result = spark.sql("SELECT dept, AVG(salary) as avg_salary FROM employees GROUP BY dept")
    print("SQL Result (still a DataFrame):")
    sql_result.show()
    
    print("\n1.2 Chain DataFrame methods on SQL result:")
    print("-" * 40)
    print("sql_result.filter(col('avg_salary') > 5000).show()")
    sql_result.filter(col("avg_salary") > 5000).show()
    
    print("\n1.3 Register DataFrame as Temporary View:")
    print("-" * 40)
    sales_people_df = spark.sql("SELECT * FROM employees WHERE dept = 'Sales'")
    sales_people_df.createOrReplaceTempView("sales_people")
    
    print("Created view 'sales_people' with Sales department employees")
    print("Query temporary view: SELECT * FROM sales_people WHERE salary > 5100")
    spark.sql("SELECT * FROM sales_people WHERE salary > 5100").show()
    
    print("\n1.4 Global Temporary Views:")
    print("-" * 40)
    employees_df.createGlobalTempView("global_employees")
    
    print("Query global temp view (note database prefix):")
    print("SELECT * FROM global_temp.global_employees LIMIT 3")
    spark.sql("SELECT * FROM global_temp.global_employees LIMIT 3").show()
    
    print("\n1.5 Clean up views:")
    print("-" * 40)
    spark.catalog.dropTempView("sales_people")
    print("Temporary view 'sales_people' dropped.")

def part2_nulls_in_spark(spark):
    """Part 2: Understanding NULLs in Spark SQL"""
    print("\n" + "="*80)
    print("PART 2: NULL Handling in Spark SQL")
    print("="*80)
    
    print("\n2.1 NULL comparison:")
    print("-" * 40)
    print("Query with equality comparison: SELECT * FROM employees WHERE dept = NULL")
    spark.sql("SELECT * FROM employees WHERE dept = NULL").show()
    
    print("\nQuery with IS NULL: SELECT * FROM employees WHERE dept IS NULL")
    spark.sql("SELECT * FROM employees WHERE dept IS NULL").show()
    
    print("\n2.2 NULL in aggregate functions:")
    print("-" * 40)
    print("Query showing how NULLs affect COUNT and AVG:")
    spark.sql("""
        SELECT 
            COUNT(*) AS total_rows,
            COUNT(dept) AS dept_count,
            COUNT(DISTINCT dept) AS unique_depts,
            AVG(salary) AS avg_salary_all,
            AVG(COALESCE(salary, 0)) AS avg_salary_with_nulls
        FROM employees
    """).show()

def part3_group_by_mechanics(spark):
    """Part 3: GROUP BY Mechanics"""
    print("\n" + "="*80)
    print("PART 3: GROUP BY Mechanics")
    print("="*80)
    
    print("\n3.1 Basic GROUP BY:")
    print("-" * 40)
    print("Before GROUP BY: individual rows")
    spark.sql("SELECT id, name, dept, salary FROM employees ORDER BY dept").show()
    
    print("\nAfter GROUP BY dept: collapsed groups")
    spark.sql("""
        SELECT dept, 
               COUNT(*) as emp_count, 
               AVG(salary) as avg_salary,
               MIN(salary) as min_salary,
               MAX(salary) as max_salary
        FROM employees
        GROUP BY dept
        ORDER BY avg_salary DESC
    """).show()
    
    print("\n3.2 WHERE vs HAVING:")
    print("-" * 40)
    print("WHERE filters rows BEFORE grouping:")
    print("Query: Filter salary > 4500 first, then group")
    spark.sql("""
        SELECT dept, COUNT(*) as count, AVG(salary) as avg_sal
        FROM employees
        WHERE salary > 4500
        GROUP BY dept
        ORDER BY avg_sal DESC
    """).show()
    
    print("\nHAVING filters groups AFTER aggregation:")
    print("Query: Filter groups with count > 1 after aggregation")
    spark.sql("""
        SELECT dept, COUNT(*) as count, AVG(salary) as avg_sal
        FROM employees
        WHERE salary > 4500
        GROUP BY dept
        HAVING COUNT(*) > 1
        ORDER BY avg_sal DESC
    """).show()

def part4_join_algorithms(spark):
    """Part 4: JOIN Algorithms"""
    print("\n" + "="*80)
    print("PART 4: JOIN Algorithms and Types")
    print("="*80)
    
    print("\n4.1 EXPLAIN join execution plan:")
    print("-" * 40)
    explain_query = """
        EXPLAIN FORMATTED
        SELECT e.*, d.location
        FROM employees e
        JOIN departments d ON e.dept = d.dept_name
    """
    print("Execution plan for JOIN:")
    explain_result = spark.sql(explain_query)
    
    for row in explain_result.collect():
        plan_text = row["plan"]
        if "== Physical Plan ==" in plan_text:
            print("\nPhysical Plan (actual execution steps):")
            lines = plan_text.split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['Project', 'Filter', 'Join', 'Exchange', 'Sort']):
                    print(line.strip())
            break
    
    print("\n4.2 Join Types:")
    print("-" * 40)
    
    joins = {
        "INNER JOIN": """
            SELECT e.name, e.dept, d.location
            FROM employees e
            INNER JOIN departments d ON e.dept = d.dept_name
            ORDER BY e.dept
        """,
        "LEFT JOIN": """
            SELECT e.name, e.dept, d.location
            FROM employees e
            LEFT JOIN departments d ON e.dept = d.dept_name
            ORDER BY e.dept
        """,
        "FULL OUTER JOIN": """
            SELECT e.name, e.dept, d.dept_name, d.location
            FROM employees e
            FULL OUTER JOIN departments d ON e.dept = d.dept_name
            ORDER BY COALESCE(e.dept, d.dept_name)
        """,
        "LEFT SEMI JOIN": """
            SELECT e.*
            FROM employees e
            LEFT SEMI JOIN departments d ON e.dept = d.dept_name
            ORDER BY e.dept
        """,
        "LEFT ANTI JOIN": """
            SELECT e.*
            FROM employees e
            LEFT ANTI JOIN departments d ON e.dept = d.dept_name
        """
    }
    
    for join_type, query in joins.items():
        print(f"\n{join_type}:")
        spark.sql(query).show()

def part5_case_expressions(spark):
    """Part 5: CASE Expression Logic Flow"""
    print("\n" + "="*80)
    print("PART 5: CASE Expressions")
    print("="*80)
    
    print("\n5.1 CASE with WHEN/THEN:")
    print("-" * 40)
    print("Salary bands with CASE expression:")
    spark.sql("""
        SELECT 
            name,
            salary,
            CASE 
                WHEN salary < 4500 THEN 'Low'
                WHEN salary < 5500 THEN 'Medium'
                WHEN salary >= 5500 THEN 'High'
                ELSE 'Unknown'
            END AS salary_band,
            CASE 
                WHEN dept IN ('Sales', 'Marketing') THEN 'Business'
                WHEN dept = 'Engineering' THEN 'Technical'
                ELSE 'Other'
            END AS dept_category
        FROM employees
        ORDER BY salary DESC
    """).show()
    
    print("\n5.2 CASE without ELSE (returns NULL):")
    print("-" * 40)
    print("Query showing what happens without ELSE clause:")
    spark.sql("""
        SELECT 
            name,
            salary,
            CASE 
                WHEN salary < 4500 THEN 'Low'
                WHEN salary < 5500 THEN 'Medium'
            END AS incomplete_band
        FROM employees
        WHERE salary >= 5500 OR salary IS NULL
    """).show()

def part6_string_array_functions(spark):
    """Part 6: String and Array Functions"""
    print("\n" + "="*80)
    print("PART 6: String and Array Functions")
    print("="*80)
    
    print("\n6.1 String Functions (1-based indexing):")
    print("-" * 40)
    print("SUBSTRING uses 1-based positions:")
    spark.sql("""
        SELECT 
            name,
            SUBSTRING(name, 1, 3) AS first_three,
            SUBSTRING(name, 2, 3) AS chars_2_to_4,
            UPPER(name) AS upper_name,
            LENGTH(name) AS name_length
        FROM employees
        LIMIT 5
    """).show()
    
    print("\n6.2 Array Functions - Indexing methods:")
    print("-" * 40)
    print("Spark SQL provides two ways to access array elements:")
    print("1. items[0] - 0-based indexing, requires bounds checking")
    print("2. GET(items, 1) - 1-based indexing, returns NULL for invalid indices")
    
    print("\nArray data for demonstration:")
    spark.sql("SELECT * FROM arrays ORDER BY id").show()
    
    print("\nArray access with bracket notation (0-based):")
    spark.sql("""
        SELECT 
            id,
            items,
            CASE 
                WHEN SIZE(items) > 0 THEN items[0]
                ELSE NULL 
            END AS first_item_bracket,
            CASE 
                WHEN SIZE(items) > 1 THEN items[1]
                ELSE NULL 
            END AS second_item_bracket
        FROM arrays
        ORDER BY id
    """).show()
    
    print("\nArray access with GET() function (1-based):")
    spark.sql("""
        SELECT 
            id,
            items,
            GET(items, 1) AS first_item_get,
            GET(items, 2) AS second_item_get,
            GET(items, 10) AS tenth_item_get
        FROM arrays
        ORDER BY id
    """).show()
    
    print("\n6.3 Array manipulation with employee skills:")
    print("-" * 40)
    spark.sql("""
        SELECT 
            name,
            skills,
            SIZE(skills) AS skill_count,
            ARRAY_CONTAINS(skills, 'Python') AS knows_python,
            ARRAY_CONTAINS(skills, 'Java') AS knows_java,
            SLICE(skills, 1, 2) AS first_two_skills
        FROM employees
        ORDER BY name
    """).show()

def part7_ctes_vs_subqueries(spark):
    """Part 7: CTEs vs Subqueries"""
    print("\n" + "="*80)
    print("PART 7: CTEs vs Subqueries")
    print("="*80)
    
    print("\n7.1 CTE Version (Recommended for readability):")
    print("-" * 40)
    print("WITH filtered_employees AS (...), department_averages AS (...) ...")
    spark.sql("""
        WITH 
        filtered_employees AS (
            SELECT * FROM employees WHERE salary > 4500
        ),
        department_averages AS (
            SELECT dept, AVG(salary) AS avg_salary 
            FROM filtered_employees 
            GROUP BY dept
        ),
        final_result AS (
            SELECT 
                da.*,
                RANK() OVER (ORDER BY da.avg_salary DESC) AS dept_rank
            FROM department_averages da
        )
        SELECT * FROM final_result ORDER BY dept_rank
    """).show()
    
    print("\n7.2 Nested Subquery Version (harder to read):")
    print("-" * 40)
    spark.sql("""
        SELECT 
            dept,
            avg_salary,
            RANK() OVER (ORDER BY avg_salary DESC) AS dept_rank
        FROM (
            SELECT dept, AVG(salary) AS avg_salary
            FROM (
                SELECT * FROM employees WHERE salary > 4500
            ) filtered
            GROUP BY dept
        ) aggregated
        ORDER BY dept_rank
    """).show()

def part8_union_vs_unionall(spark):
    """Part 8: UNION vs UNION ALL"""
    print("\n" + "="*80)
    print("PART 8: UNION vs UNION ALL")
    print("="*80)
    
    print("\n8.1 Original data:")
    print("-" * 40)
    print("Q1 sales data:")
    spark.sql("SELECT * FROM sales_q1").show()
    print("\nQ2 sales data:")
    spark.sql("SELECT * FROM sales_q2").show()
    
    print("\n8.2 UNION ALL (keeps all rows):")
    print("-" * 40)
    print("Concatenates all rows from both datasets:")
    spark.sql("""
        SELECT * FROM sales_q1
        UNION ALL
        SELECT * FROM sales_q2
    """).show()
    
    print("\n8.3 UNION (removes duplicate rows):")
    print("-" * 40)
    print("Removes duplicate rows after concatenation:")
    spark.sql("""
        SELECT * FROM sales_q1
        UNION
        SELECT * FROM sales_q2
    """).show()
    
    print("\n8.4 Alternative deduplication pattern:")
    print("-" * 40)
    print("UNION ALL combined with DISTINCT:")
    spark.sql("""
        SELECT DISTINCT * FROM (
            SELECT * FROM sales_q1
            UNION ALL
            SELECT * FROM sales_q2
        )
    """).show()

def part9_explain_output(spark):
    """Part 9: Understanding EXPLAIN Output"""
    print("\n" + "="*80)
    print("PART 9: EXPLAIN Output Analysis")
    print("="*80)
    
    print("\n9.1 Simple EXPLAIN example:")
    print("-" * 40)
    print("Query: SELECT dept, AVG(salary) FROM employees GROUP BY dept")
    spark.sql("EXPLAIN SELECT dept, AVG(salary) FROM employees GROUP BY dept").show(truncate=False)
    
    print("\n9.2 Key components in EXPLAIN output:")
    print("-" * 40)
    print("""
    • HashAggregate or SortAggregate: GROUP BY operations
    • Exchange: Data shuffling between nodes
    • Filter: Row filtering operations
    • Scan: Data source reading
    • Project: Column selection
    """)

def part10_lateral_view_explode(spark):
    """Part 10: Lateral View with Explode"""
    print("\n" + "="*80)
    print("PART 10: LATERAL VIEW with EXPLODE")
    print("="*80)
    
    print("\n10.1 Original arrays data:")
    print("-" * 40)
    spark.sql("SELECT * FROM arrays").show()
    
    print("\n10.2 LATERAL VIEW EXPLODE:")
    print("-" * 40)
    print("Flattens arrays into individual rows:")
    spark.sql("""
        SELECT 
            a.id,
            exploded.item
        FROM arrays a
        LATERAL VIEW EXPLODE(a.items) exploded AS item
        ORDER BY a.id, item
    """).show()
    
    print("\n10.3 Employee skills example:")
    print("-" * 40)
    spark.sql("""
        SELECT 
            e.name,
            e.dept,
            exploded.skill
        FROM employees e
        LATERAL VIEW EXPLODE(e.skills) exploded AS skill
        ORDER BY e.name, exploded.skill
    """).show()

def part11_window_functions(spark):
    """Part 11: Window Functions vs GROUP BY"""
    print("\n" + "="*80)
    print("PART 11: Window Functions vs GROUP BY")
    print("="*80)
    
    print("\n11.1 GROUP BY (collapses rows):")
    print("-" * 40)
    print("One row per department:")
    spark.sql("""
        SELECT dept, 
               COUNT(*) as emp_count,
               AVG(salary) as avg_salary
        FROM employees
        WHERE dept IS NOT NULL
        GROUP BY dept
        ORDER BY avg_salary DESC
    """).show()
    
    print("\n11.2 Window Functions (preserves rows):")
    print("-" * 40)
    print("All rows preserved, with aggregate values added:")
    spark.sql("""
        SELECT 
            name,
            dept,
            salary,
            AVG(salary) OVER (PARTITION BY dept) AS dept_avg_salary,
            salary - AVG(salary) OVER (PARTITION BY dept) AS diff_from_avg,
            RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS dept_rank
        FROM employees
        WHERE dept IS NOT NULL
        ORDER BY dept, salary DESC
    """).show()
    
    print("\n11.3 Running total with Window:")
    print("-" * 40)
    spark.sql("""
        SELECT 
            name,
            salary,
            SUM(salary) OVER (ORDER BY salary ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total,
            ROUND(100.0 * salary / SUM(salary) OVER (), 2) AS percent_of_total
        FROM employees
        WHERE salary IS NOT NULL
        ORDER BY salary
    """).show()

def part12_broadcast_hint(spark):
    """Part 12: Broadcast Hint"""
    print("\n" + "="*80)
    print("PART 12: Broadcast Hint for Joins")
    print("="*80)
    
    print("\n12.1 Join without hint (Spark decides algorithm):")
    print("-" * 40)
    spark.sql("""
        SELECT e.name, e.dept, d.location
        FROM employees e
        JOIN departments d ON e.dept = d.dept_name
        ORDER BY e.dept
        LIMIT 5
    """).show()
    
    print("\n12.2 Join with BROADCAST hint:")
    print("-" * 40)
    print("Explicit broadcast join for small 'departments' table:")
    spark.sql("""
        SELECT /*+ BROADCAST(d) */ 
            e.name,
            e.dept,
            d.location,
            d.budget
        FROM employees e
        JOIN departments d ON e.dept = d.dept_name
        ORDER BY e.dept
        LIMIT 5
    """).show()

def part13_anti_patterns(spark):
    """Part 13: Common Anti-Patterns"""
    print("\n" + "="*80)
    print("PART 13: Common Anti-Patterns to Avoid")
    print("="*80)
    
    print("\n13.1 ANTI-PATTERN: SELECT *")
    print("-" * 40)
    print("Instead of: SELECT * FROM employees")
    print("Use specific columns: SELECT id, name, dept FROM employees WHERE dept = 'Sales'")
    
    spark.sql("SELECT id, name, dept FROM employees WHERE dept = 'Sales'").show()
    
    print("\n13.2 ANTI-PATTERN: Functions in JOIN condition")
    print("-" * 40)
    print("Instead of: JOIN ON UPPER(t1.name) = UPPER(t2.name)")
    print("Pre-compute transformations in CTE first")

def part14_debugging_workflow(spark):
    """Part 14: Practical Debugging"""
    print("\n" + "="*80)
    print("PART 14: Practical Debugging Workflow")
    print("="*80)
    
    print("\n14.1 Check data quality and NULLs:")
    print("-" * 40)
    checks = [
        "SELECT COUNT(*) AS total_rows FROM employees",
        "SELECT COUNT(DISTINCT dept) AS unique_depts FROM employees",
        "SELECT dept, COUNT(*) AS count FROM employees GROUP BY dept ORDER BY count DESC",
        "SELECT SUM(CASE WHEN salary IS NULL THEN 1 ELSE 0 END) AS null_salaries FROM employees"
    ]
    
    for i, check in enumerate(checks, 1):
        print(f"\nCheck {i}: {check}")
        spark.sql(check).show()
    
    print("\n14.2 Check join key distribution:")
    print("-" * 40)
    spark.sql("""
        SELECT 
            'Employees' as source, dept, COUNT(*) as count
        FROM employees 
        WHERE dept IS NOT NULL
        GROUP BY dept
        UNION ALL
        SELECT 
            'Departments' as source, dept_name as dept, COUNT(*) as count
        FROM departments
        GROUP BY dept_name
        ORDER BY dept, source
    """).show()

def part15_essential_patterns(spark):
    """Part 15: Essential Patterns"""
    print("\n" + "="*80)
    print("PART 15: Essential Patterns to Memorize")
    print("="*80)
    
    print("\n15.1 Fill NULLs with COALESCE:")
    print("-" * 40)
    spark.sql("""
        SELECT 
            name,
            dept,
            COALESCE(dept, 'Unknown') AS dept_filled,
            salary,
            COALESCE(salary, 0) AS salary_filled
        FROM employees
        ORDER BY dept
    """).show()
    
    print("\n15.2 Safe division with NULLIF:")
    print("-" * 40)
    spark.sql("""
        SELECT 
            name,
            salary,
            CASE 
                WHEN salary > 0 THEN 100000.0 / salary 
                ELSE NULL 
            END AS revenue_per_salary,
            100000.0 / NULLIF(salary, 0) AS safe_revenue_per_salary
        FROM employees
        WHERE salary IS NOT NULL
        ORDER BY salary DESC
        LIMIT 5
    """).show()
    
    print("\n15.3 Find latest per group (using Window):")
    print("-" * 40)
    transactions_data = [
        (1, "2023-01-01", 100),
        (1, "2023-01-02", 200),
        (2, "2023-01-01", 150),
        (2, "2023-01-03", 250),
        (1, "2023-01-03", 300)
    ]
    
    transactions_df = spark.createDataFrame(transactions_data, ["cust_id", "date", "amount"])
    transactions_df.createOrReplaceTempView("transactions")
    
    print("Sample transaction data:")
    spark.sql("SELECT * FROM transactions ORDER BY cust_id, date").show()
    
    print("\nFind latest transaction per customer:")
    spark.sql("""
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY cust_id ORDER BY date DESC) as rn
            FROM transactions
        )
        SELECT cust_id, date, amount
        FROM ranked
        WHERE rn = 1
        ORDER BY cust_id
    """).show()
    
    print("\n15.4 Pivot with CASE (when PIVOT not available):")
    print("-" * 40)
    spark.sql("""
        SELECT 
            dept,
            SUM(CASE WHEN salary < 4500 THEN 1 ELSE 0 END) AS low_count,
            SUM(CASE WHEN salary BETWEEN 4500 AND 5500 THEN 1 ELSE 0 END) AS medium_count,
            SUM(CASE WHEN salary > 5500 THEN 1 ELSE 0 END) AS high_count,
            COUNT(*) AS total
        FROM employees
        WHERE dept IS NOT NULL
        GROUP BY dept
        ORDER BY dept
    """).show()

def run_demo_without_errors():
    """Run a robust demo that handles all potential errors"""
    print("Starting Spark SQL Teaching Demo...")
    print("="*80)
    
    # Initialize Spark
    spark = setup_spark_session()
    
    try:
        # Create sample data
        print("Creating sample datasets...")
        dataframes = create_sample_data(spark)
        
        # Register all DataFrames as views BEFORE starting demo
        register_initial_views(spark, dataframes)
        
        # Run key parts of the demo
        parts_to_run = [
            ("Part 1: SQL Execution", lambda: part1_sql_execution(spark, dataframes)),
            ("Part 2: NULL Handling", lambda: part2_nulls_in_spark(spark)),
            ("Part 3: GROUP BY Mechanics", lambda: part3_group_by_mechanics(spark)),
            ("Part 4: JOIN Algorithms", lambda: part4_join_algorithms(spark)),
            ("Part 5: CASE Expressions", lambda: part5_case_expressions(spark)),
            ("Part 6: String/Array Functions", lambda: part6_string_array_functions(spark)),
            ("Part 7: CTEs vs Subqueries", lambda: part7_ctes_vs_subqueries(spark)),
            ("Part 8: UNION vs UNION ALL", lambda: part8_union_vs_unionall(spark)),
            ("Part 9: EXPLAIN Output", lambda: part9_explain_output(spark)),
            ("Part 10: LATERAL VIEW EXPLODE", lambda: part10_lateral_view_explode(spark)),
            ("Part 11: Window Functions", lambda: part11_window_functions(spark)),
            ("Part 12: Broadcast Hint", lambda: part12_broadcast_hint(spark)),
            ("Part 13: Anti-Patterns", lambda: part13_anti_patterns(spark)),
            ("Part 14: Debugging", lambda: part14_debugging_workflow(spark)),
            ("Part 15: Essential Patterns", lambda: part15_essential_patterns(spark)),
        ]
        
        for i, (name, func) in enumerate(parts_to_run, 1):
            print(f"\n{'='*80}")
            print(f"Running {name}...")
            print(f"{'='*80}")
            try:
                func()
                print(f"{name} completed successfully")
            except Exception as e:
                print(f"Error in {name}: {str(e)[:100]}...")
                print("Continuing with next part...")
            
            if i < len(parts_to_run):
                input(f"\nPress Enter to continue to Part {i+1}...")
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nFatal error during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        spark.stop()
        print("\nSpark session stopped. Demo complete!")

if __name__ == "__main__":
    run_demo_without_errors()