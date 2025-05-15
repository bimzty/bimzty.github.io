SELECT * FROM Sale;

EXECUTE sale_operations.cancel_sale_record(1);

SELECT * FROM Sale;

ROLLBACK;
