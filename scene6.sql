-- With index, this should show INDEX RANGE SCAN
EXPLAIN PLAN FOR
SELECT * FROM option_list WHERE option_category = 'electrical';

SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY);

-- Without index (force full table scan), this should show Table Access Full
EXPLAIN PLAN FOR
SELECT /*+ FULL(option_list) */ * FROM option_list WHERE option_category = 'electrical';

SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY);
