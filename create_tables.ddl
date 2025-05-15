-- Drop tables and associated data
DROP TABLE task CASCADE CONSTRAINTS;
-- DROP TABLE stylizes CASCADE CONSTRAINTS;
DROP TABLE describes CASCADE CONSTRAINTS;
DROP TABLE subdivision CASCADE CONSTRAINTS;
DROP TABLE school_district CASCADE CONSTRAINTS;
DROP TABLE school CASCADE CONSTRAINTS;
DROP TABLE sales_representative CASCADE CONSTRAINTS;
DROP TABLE sale CASCADE CONSTRAINTS;
DROP TABLE room CASCADE CONSTRAINTS;
DROP TABLE option_list CASCADE CONSTRAINTS;
DROP TABLE lot CASCADE CONSTRAINTS;
DROP TABLE house_style CASCADE CONSTRAINTS;
DROP TABLE escrow CASCADE CONSTRAINTS;
DROP TABLE employee CASCADE CONSTRAINTS;
DROP TABLE elevation CASCADE CONSTRAINTS;
DROP TABLE decorator_choice CASCADE CONSTRAINTS;
DROP TABLE customer CASCADE CONSTRAINTS;
DROP TABLE construction_project CASCADE CONSTRAINTS;
DROP TABLE construction_manager CASCADE CONSTRAINTS;
DROP TABLE chosen_style CASCADE CONSTRAINTS;
DROP TABLE bank CASCADE CONSTRAINTS;
-- DROP TRIGGER arc_fkarc_sales_representative;
-- DROP TRIGGER arc_fkarc_construction_manager;


CREATE TABLE bank (
    bank_id   NUMBER(10) NOT NULL,
    phone_no  NUMBER(10),
    fax_no    NUMBER(10),
    street    VARCHAR2(30),
    city      VARCHAR2(20),
    state_zip NUMBER(5)
);

ALTER TABLE bank ADD CONSTRAINT bank_pk PRIMARY KEY ( bank_id );

CREATE TABLE chosen_style (
    schoice_id             NUMBER(10) NOT NULL,
    reversed_style         CHAR(1) NOT NULL,
    elevation_chosen       NUMBER(10) NOT NULL,
    house_style_style_name VARCHAR2(20) NOT NULL,
    sale_invoice_id        NUMBER(10) NOT NULL
);

CREATE UNIQUE INDEX chosen_style__idx ON
    chosen_style (
        sale_invoice_id
    ASC );

ALTER TABLE chosen_style ADD CONSTRAINT chosen_style_pk PRIMARY KEY ( schoice_id );

CREATE TABLE construction_manager (
    employee_id NUMBER(10) NOT NULL,
    crew_no     NUMBER(10)
);

ALTER TABLE construction_manager ADD CONSTRAINT construction_manager_pk PRIMARY KEY ( employee_id );

CREATE TABLE construction_project (
    project_id            NUMBER(10) NOT NULL,
    start_date            DATE NOT NULL,
    est_completion_date   DATE,
    current_stage         NUMBER(1) NOT NULL,
    project_pct_complete  NUMBER(5, 2) DEFAULT 0 NOT NULL,
    construction_photo    httpuritype,
    -- sale_invoice_id       NUMBER(10) NOT NULL,
    cm_employee_id        NUMBER(10) NOT NULL
    -- project_pct_complete1 NUMBER(5, 2) DEFAULT 0
);

ALTER TABLE construction_project
    ADD CHECK ( current_stage IN ( 1, 2, 3, 4, 5, 6, 7 ) );

-- CREATE UNIQUE INDEX construction_project__idx ON
--     construction_project (
--         sale_invoice_id
--     ASC );

ALTER TABLE construction_project ADD CONSTRAINT construction_project_pk PRIMARY KEY ( project_id );

CREATE TABLE customer (
    customer_id    NUMBER(10) NOT NULL,
    customer_fname VARCHAR2(20) NOT NULL,
    customer_lname VARCHAR2(20) NOT NULL,
    street         VARCHAR2(30),
    city           VARCHAR2(20),
    state_zip      NUMBER(5)
);

ALTER TABLE customer ADD CONSTRAINT customer_pk PRIMARY KEY ( customer_id );

CREATE TABLE decorator_choice (
    dchoice_id                  NUMBER(10) NOT NULL,
    choice_selected_date        DATE NOT NULL,
    item_quantity               NUMBER(3) NOT NULL,
    current_opt_price           NUMBER(10, 2) NOT NULL,
    option_list_option_id       NUMBER(10) NOT NULL,
    option_list_stage_no        NUMBER(1) NOT NULL,
    option_list_style_name      VARCHAR2(20) NOT NULL,
    cp_project_id               NUMBER(10) NOT NULL,
    room_house_style_style_name VARCHAR2(20),
    room_room_id                NUMBER(10)
);

ALTER TABLE decorator_choice
    ADD CHECK ( option_list_stage_no IN ( 1, 2, 3, 4, 5, 6, 7 ) );

ALTER TABLE decorator_choice ADD CONSTRAINT decorator_choice_pk PRIMARY KEY ( dchoice_id );

CREATE TABLE elevation (
    elevation_id           NUMBER(10) NOT NULL,
    elevation_desc         VARCHAR2(200),
    elevation_cost         NUMBER(10, 2) NOT NULL,
    house_style_style_name VARCHAR2(20) NOT NULL
);

ALTER TABLE elevation ADD CONSTRAINT elevation_pk PRIMARY KEY ( elevation_id,
                                                                house_style_style_name );

ALTER TABLE elevation ADD CONSTRAINT style_name_elevation_id_un UNIQUE ( house_style_style_name,
                                                                         elevation_id );

CREATE TABLE employee (
    employee_id    NUMBER(10) NOT NULL,
    employee_fname VARCHAR2(20) NOT NULL,
    employee_lname VARCHAR2(20) NOT NULL,
    title          VARCHAR2(30) NOT NULL
);

ALTER TABLE employee
    ADD CONSTRAINT ch_inh_employee CHECK ( title IN ( 'Construction_Manager', 'Employee', 'Sales_Representative' ) );

ALTER TABLE employee ADD CONSTRAINT employee_pk PRIMARY KEY ( employee_id );

CREATE TABLE escrow (
    escrow_id       NUMBER(10) NOT NULL,
    escrow_agent_id NUMBER(10) NOT NULL,
    deposit_amount  NUMBER(10, 2) NOT NULL,
    street          VARCHAR2(30),
    city            VARCHAR2(20),
    state_zip       NUMBER(5)
    -- sale_invoice_id NUMBER(10) NOT NULL
);

-- CREATE UNIQUE INDEX escrow__idx ON
--     escrow (
--         sale_invoice_id
--     ASC );

ALTER TABLE escrow ADD CONSTRAINT escrow_pk PRIMARY KEY ( escrow_id );

CREATE TABLE house_style (
    style_name        VARCHAR2(20) NOT NULL,
    house_style_photo httpuritype,
    base_price        NUMBER(10, 2) NOT NULL,
    style_desc        VARCHAR2(200),
    style_size        NUMBER(7, 2) DEFAULT 0 NOT NULL,
    floor_plan_link   httpuritype
    -- style_size1       NUMBER(7, 2) NOT NULL
);

ALTER TABLE house_style ADD CONSTRAINT house_style_pk PRIMARY KEY ( style_name );

CREATE TABLE lot (
    lot_id             NUMBER(10) NOT NULL,
    street             VARCHAR2(30) NOT NULL,
    city               VARCHAR2(20) NOT NULL,
    state_zip          NUMBER(5) NOT NULL,
    latitude           NUMBER(7) NOT NULL,
    longitude         NUMBER(7) NOT NULL,
    subdivision_sub_id NUMBER(10) NOT NULL
);

ALTER TABLE lot ADD CONSTRAINT lot_pk PRIMARY KEY ( lot_id );

CREATE TABLE option_list (
    option_id       NUMBER(10) NOT NULL,
    stage_no        NUMBER(1) NOT NULL,
    style_name      VARCHAR2(20) NOT NULL,
    option_cost     NUMBER(10, 2) NOT NULL,
    option_category VARCHAR2(20) NOT NULL,
    option_desc     VARCHAR2(200)
);

ALTER TABLE option_list
    ADD CHECK ( stage_no IN ( 1, 2, 3, 4, 5, 6, 7 ) );

ALTER TABLE option_list
    ADD CHECK ( option_category IN ( 'Electrical', 'Exterior', 'Interior', 'Plumbing' ) );

ALTER TABLE option_list
    ADD CONSTRAINT option_list_pk PRIMARY KEY ( option_id,
                                                stage_no,
                                                style_name );

CREATE TABLE room (
    room_id                NUMBER(10) NOT NULL,
    room_name              VARCHAR2(20) NOT NULL,
    room_size              NUMBER(7, 2) NOT NULL,
    floor                  NUMBER(2) NOT NULL,
    room_desc              VARCHAR2(200),
    window_no              NUMBER(2),
    ceiling                VARCHAR2(50),
    house_style_style_name VARCHAR2(20) NOT NULL
);

ALTER TABLE room ADD CONSTRAINT room_pk PRIMARY KEY ( room_id,
                                                      house_style_style_name );

ALTER TABLE room ADD CONSTRAINT style_name_room_id_un UNIQUE ( house_style_style_name,
                                                               room_id );

CREATE TABLE sale (
    invoice_id              NUMBER(10) NOT NULL,
    lot_premium             NUMBER(10, 2) DEFAULT 0 NOT NULL,
    date_sold               DATE NOT NULL,
    financing_method        VARCHAR2(20) NOT NULL,
    -- lot_size                NUMBER(7, 2),
    -- contract_status         VARCHAR2(10) DEFAULT 'Valid' NOT NULL,
    lot_lot_id              NUMBER(10) NOT NULL,
    escrow_escrow_id        NUMBER(10) NOT NULL,
    cp_project_id           NUMBER(10) NOT NULL,
    sr_liscense_no          NUMBER(10) NOT NULL,
    customer_customer_id    NUMBER(10) NOT NULL,
    bank_bank_id            NUMBER(10)
    -- chosen_style_schoice_id NUMBER(10) NOT NULL
    -- lot_premium1            NUMBER(10, 2)
);

ALTER TABLE sale
    ADD CHECK ( financing_method IN ( 'Mortgage Loan', 'Seller Financing' ) );

-- ALTER TABLE sale
--     ADD CHECK ( contract_status IN ( 'Valid', 'Void' ) );

CREATE UNIQUE INDEX sale__idx ON
    sale (
        cp_project_id
    ASC );

CREATE UNIQUE INDEX sale__idxv1 ON
    sale (
        escrow_escrow_id
    ASC );

CREATE UNIQUE INDEX sale__idxv2 ON
    sale (
        lot_lot_id
    ASC );

-- CREATE UNIQUE INDEX sale__idxv3 ON
--     sale (
--         chosen_style_schoice_id
--     ASC );

ALTER TABLE sale ADD CONSTRAINT sale_pk PRIMARY KEY ( invoice_id );

CREATE TABLE sales_representative (
    employee_id NUMBER(10) NOT NULL,
    liscense_no NUMBER(10) NOT NULL
);

ALTER TABLE sales_representative ADD CONSTRAINT sales_representative_pk PRIMARY KEY ( employee_id );

ALTER TABLE sales_representative ADD CONSTRAINT sales_representative_pkv1 UNIQUE ( liscense_no );

CREATE TABLE school (
    school_name                 VARCHAR2(20) NOT NULL,
    school_level                VARCHAR2(10) NOT NULL,
    school_district_district_id NUMBER(10) NOT NULL
);

ALTER TABLE school
    ADD CHECK ( school_level IN ( 'Elementary', 'High', 'Middle' ) );

ALTER TABLE school ADD CONSTRAINT school_pk PRIMARY KEY ( school_name );

CREATE TABLE school_district (
    district_id        NUMBER(10) NOT NULL,
    district_name      VARCHAR2(20) NOT NULL,
    subdivision_sub_id NUMBER(10) NOT NULL
);

ALTER TABLE school_district ADD CONSTRAINT school_district_pk PRIMARY KEY ( district_id );

CREATE TABLE describes (
    lot_lot_id             NUMBER(10) NOT NULL,
    house_style_style_name VARCHAR2(20) NOT NULL
);

ALTER TABLE describes  ADD CONSTRAINT describes_pk PRIMARY KEY ( lot_lot_id,
                                                              house_style_style_name );

CREATE TABLE subdivision (
    sub_id   NUMBER(10) NOT NULL,
    sub_name VARCHAR2(20),
    sub_map  httpuritype
);

ALTER TABLE subdivision ADD CONSTRAINT subdivision_pk PRIMARY KEY ( sub_id );

CREATE TABLE task (
    task_id           NUMBER(10) NOT NULL,
    task_desc         VARCHAR2(200),
    task_pct_complete NUMBER(5, 2) DEFAULT 0 NOT NULL,
    cp_project_id     NUMBER(10) NOT NULL
);

ALTER TABLE chosen_style
    ADD CONSTRAINT chosen_style_house_style_fk FOREIGN KEY ( house_style_style_name )
        REFERENCES house_style ( style_name );

ALTER TABLE chosen_style
    ADD CONSTRAINT chosen_style_sale_fk FOREIGN KEY ( sale_invoice_id )
        REFERENCES sale ( invoice_id );

ALTER TABLE construction_manager
    ADD CONSTRAINT cm_employee_fk FOREIGN KEY ( employee_id )
        REFERENCES employee ( employee_id );

-- ALTER TABLE construction_project
--     ADD CONSTRAINT construction_project_sale_fk FOREIGN KEY ( sale_invoice_id )
--         REFERENCES sale ( invoice_id );

ALTER TABLE construction_project
    ADD CONSTRAINT cp_cm_fk FOREIGN KEY ( cm_employee_id )
        REFERENCES construction_manager ( employee_id );

ALTER TABLE decorator_choice
    ADD CONSTRAINT dc_cp_fk FOREIGN KEY ( cp_project_id )
        REFERENCES construction_project ( project_id );

ALTER TABLE decorator_choice
    ADD CONSTRAINT dc_option_list_fk FOREIGN KEY ( option_list_option_id,
                                                   option_list_stage_no,
                                                   option_list_style_name )
        REFERENCES option_list ( option_id,
                                 stage_no,
                                 style_name );

ALTER TABLE decorator_choice
    ADD CONSTRAINT decorator_choice_room_fk FOREIGN KEY ( room_house_style_style_name,
                                                          room_room_id )
        REFERENCES room ( house_style_style_name,
                          room_id );

ALTER TABLE elevation
    ADD CONSTRAINT elevation_house_style_fk FOREIGN KEY ( house_style_style_name )
        REFERENCES house_style ( style_name )
            ON DELETE CASCADE;

-- ALTER TABLE escrow
--     ADD CONSTRAINT escrow_sale_fk FOREIGN KEY ( sale_invoice_id )
--         REFERENCES sale ( invoice_id );

ALTER TABLE describes
    ADD CONSTRAINT hasv5_house_style_fk FOREIGN KEY ( house_style_style_name )
        REFERENCES house_style ( style_name );

ALTER TABLE describes
    ADD CONSTRAINT hasv5_lot_fk FOREIGN KEY ( lot_lot_id )
        REFERENCES lot ( lot_id );

ALTER TABLE lot
    ADD CONSTRAINT lot_subdivision_fk FOREIGN KEY ( subdivision_sub_id )
        REFERENCES subdivision ( sub_id );

ALTER TABLE room
    ADD CONSTRAINT room_house_style_fk FOREIGN KEY ( house_style_style_name )
        REFERENCES house_style ( style_name )
            ON DELETE CASCADE;

ALTER TABLE sale
    ADD CONSTRAINT sale_bank_fk FOREIGN KEY ( bank_bank_id )
        REFERENCES bank ( bank_id );

-- ALTER TABLE sale
--     ADD CONSTRAINT sale_chosen_style_fk FOREIGN KEY ( chosen_style_schoice_id )
--         REFERENCES chosen_style ( schoice_id );

ALTER TABLE sale
    ADD CONSTRAINT sale_construction_project_fk FOREIGN KEY ( cp_project_id )
        REFERENCES construction_project ( project_id );

ALTER TABLE sale
    ADD CONSTRAINT sale_customer_fk FOREIGN KEY ( customer_customer_id )
        REFERENCES customer ( customer_id );

ALTER TABLE sale
    ADD CONSTRAINT sale_escrow_fk FOREIGN KEY ( escrow_escrow_id )
        REFERENCES escrow ( escrow_id );

ALTER TABLE sale
    ADD CONSTRAINT sale_lot_fk FOREIGN KEY ( lot_lot_id )
        REFERENCES lot ( lot_id );

ALTER TABLE sale
    ADD CONSTRAINT sale_sales_representative_fk FOREIGN KEY ( sr_liscense_no )
        REFERENCES sales_representative ( liscense_no );

ALTER TABLE school_district
    ADD CONSTRAINT school_district_subdivision_fk FOREIGN KEY ( subdivision_sub_id )
        REFERENCES subdivision ( sub_id );

ALTER TABLE school
    ADD CONSTRAINT school_school_district_fk FOREIGN KEY ( school_district_district_id )
        REFERENCES school_district ( district_id );

ALTER TABLE sales_representative
    ADD CONSTRAINT sr_employee_fk FOREIGN KEY ( employee_id )
        REFERENCES employee ( employee_id );

ALTER TABLE task
    ADD CONSTRAINT task_construction_project_fk FOREIGN KEY ( cp_project_id )
        REFERENCES construction_project ( project_id );

-- CREATE OR REPLACE TRIGGER arc_fkarc_sales_representative BEFORE
--     INSERT OR UPDATE OF employee_id ON sales_representative
--     FOR EACH ROW
-- DECLARE
--     d VARCHAR2(10);
-- BEGIN
--     SELECT
--         a.title
--     INTO d
--     FROM
--         employee a
--     WHERE
--         a.employee_id = :new.employee_id;
--
--     IF ( d IS NULL OR d <> 'Sales_Representative' ) THEN
--         raise_application_error(
--                                -20223,
--                                'FK SR_Employee_FK in Table Sales_Representative violates Arc constraint on Table Employee - discriminator column title doesn''t have value ''Sales_Representative'''
--         );
--     END IF;
--
-- EXCEPTION
--     WHEN no_data_found THEN
--         NULL;
--     WHEN OTHERS THEN
--         RAISE;
-- END;
-- /
--
-- CREATE OR REPLACE TRIGGER arc_fkarc_construction_manager BEFORE
--     INSERT OR UPDATE OF employee_id ON construction_manager
--     FOR EACH ROW
-- DECLARE
--     d VARCHAR2(10);
-- BEGIN
--     SELECT
--         a.title
--     INTO d
--     FROM
--         employee a
--     WHERE
--         a.employee_id = :new.employee_id;
--
--     IF ( d IS NULL OR d <> 'Construction_Manager' ) THEN
--         raise_application_error(
--                                -20223,
--                                'FK CM_Employee_FK in Table Construction_Manager violates Arc constraint on Table Employee - discriminator column title doesn''t have value ''Construction_Manager'''
--         );
--     END IF;
--
-- EXCEPTION
--     WHEN no_data_found THEN
--         NULL;
--     WHEN OTHERS THEN
--         RAISE;
-- END;
-- /



-- Oracle SQL Developer Data Modeler Summary Report:
--
-- CREATE TABLE                            20
-- CREATE INDEX                             7
-- ALTER TABLE                             56
-- CREATE VIEW                              0
-- ALTER VIEW                               0
-- CREATE PACKAGE                           0
-- CREATE PACKAGE BODY                      0
-- CREATE PROCEDURE                         0
-- CREATE FUNCTION                          0
-- CREATE TRIGGER                           2
-- ALTER TRIGGER                            0
-- CREATE COLLECTION TYPE                   0
-- CREATE STRUCTURED TYPE                   0
-- CREATE STRUCTURED TYPE BODY              0
-- CREATE CLUSTER                           0
-- CREATE CONTEXT                           0
-- CREATE DATABASE                          0
-- CREATE DIMENSION                         0
-- CREATE DIRECTORY                         0
-- CREATE DISK GROUP                        0
-- CREATE ROLE                              0
-- CREATE ROLLBACK SEGMENT                  0
-- CREATE SEQUENCE                          0
-- CREATE MATERIALIZED VIEW                 0
-- CREATE MATERIALIZED VIEW LOG             0
-- CREATE SYNONYM                           0
-- CREATE TABLESPACE                        0
-- CREATE USER                              0
--
-- DROP TABLESPACE                          0
-- DROP DATABASE                            0
--
-- REDACTION POLICY                         0
--
-- ORDS DROP SCHEMA                         0
-- ORDS ENABLE SCHEMA                       0
-- ORDS ENABLE OBJECT                       0
--
-- ERRORS                                   0
-- WARNINGS                                 0
