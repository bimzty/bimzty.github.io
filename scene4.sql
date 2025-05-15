SELECT * FROM Sale;
/
BEGIN
    create_sale_record(
        8, --customer id,
        4,--sr_liscense
        2, --bank id
        10, --lot id
        'Mortgage Loan', -- financing,
        3, -- escrow agent id
        50000, -- deposit amount
        'N',--reverse style?
        2, -- Elevation
        'Colonial', -- house style
        30  -- construction_manager employee_id
    );
END;
/

SELECT * FROM Sale;
