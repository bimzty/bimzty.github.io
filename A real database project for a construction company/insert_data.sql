/*****************************
Insert into Subdivision table
*****************************/
SELECT
    sub_id,
    sub_name,
    TREAT(sub_map AS HTTPURITYPE).GETURL() as SUB_MAP_URL
FROM subdivision;

DESC subdivision;

INSERT INTO subdivision (SUB_ID, SUB_NAME, SUB_MAP)
VALUES (subdivision_id_seq.NEXTVAL, 'Oakridge Heights', httpuritype('http://example.com/maps/oakridge-heights'));

INSERT INTO subdivision (SUB_ID, SUB_NAME, SUB_MAP)
VALUES (subdivision_id_seq.NEXTVAL, 'Willow Creek', httpuritype('http://example.com/maps/willow-creek'));

INSERT INTO subdivision (SUB_ID, SUB_NAME, SUB_MAP)
VALUES (subdivision_id_seq.NEXTVAL, 'Meadowbrook Estates', httpuritype('http://example.com/maps/meadowbrook-estates'));

INSERT INTO subdivision (SUB_ID, SUB_NAME, SUB_MAP)
VALUES (subdivision_id_seq.NEXTVAL, 'Sunnyside Gardens', httpuritype('http://example.com/maps/sunnyside-gardens'));

INSERT INTO subdivision (SUB_ID, SUB_NAME, SUB_MAP)
VALUES (subdivision_id_seq.NEXTVAL, 'Riverstone District', httpuritype('http://example.com/maps/riverstone-district'));

INSERT INTO subdivision (SUB_ID, SUB_NAME, SUB_MAP)
VALUES (subdivision_id_seq.NEXTVAL, 'Pinecrest Valley', httpuritype('http://example.com/maps/pinecrest-valley'));

INSERT INTO subdivision (SUB_ID, SUB_NAME, SUB_MAP)
VALUES (subdivision_id_seq.NEXTVAL, 'Maplewood Commons', httpuritype('http://example.com/maps/maplewood-commons'));

COMMIT;

/*****************************
Insert into School_District table
*****************************/

INSERT INTO school_district (DISTRICT_ID, DISTRICT_NAME, SUBDIVISION_SUB_ID)
VALUES (district_id_seq.NEXTVAL, 'Oakridge Unified', 1);

INSERT INTO school_district (DISTRICT_ID, DISTRICT_NAME, SUBDIVISION_SUB_ID)
VALUES (district_id_seq.NEXTVAL, 'Willow Creek Edu', 2);

INSERT INTO school_district (DISTRICT_ID, DISTRICT_NAME, SUBDIVISION_SUB_ID)
VALUES (district_id_seq.NEXTVAL, 'Meadowbrook Schools', 3);

INSERT INTO school_district (DISTRICT_ID, DISTRICT_NAME, SUBDIVISION_SUB_ID)
VALUES (district_id_seq.NEXTVAL, 'Riverstone District', 5);

INSERT INTO school_district (DISTRICT_ID, DISTRICT_NAME, SUBDIVISION_SUB_ID)
VALUES (district_id_seq.NEXTVAL, 'Pinecrest Valley Edu', 6);

COMMIT;

/*****************************
Insert into School table
*****************************/

INSERT INTO school (SCHOOL_NAME, SCHOOL_LEVEL, SCHOOL_DISTRICT_DISTRICT_ID)
VALUES ('Oak Elementary', 'Elementary', 1);

INSERT INTO school (SCHOOL_NAME, SCHOOL_LEVEL, SCHOOL_DISTRICT_DISTRICT_ID)
VALUES ('Oak High', 'High', 1);

INSERT INTO school (SCHOOL_NAME, SCHOOL_LEVEL, SCHOOL_DISTRICT_DISTRICT_ID)
VALUES ('Willow Creek', 'Middle', 2);

INSERT INTO school (SCHOOL_NAME, SCHOOL_LEVEL, SCHOOL_DISTRICT_DISTRICT_ID)
VALUES ('Creekside Elem', 'Elementary', 2);

INSERT INTO school (SCHOOL_NAME, SCHOOL_LEVEL, SCHOOL_DISTRICT_DISTRICT_ID)
VALUES ('Meadowbrook High', 'High', 3);

INSERT INTO school (SCHOOL_NAME, SCHOOL_LEVEL, SCHOOL_DISTRICT_DISTRICT_ID)
VALUES ('Brook Elementary', 'Elementary', 3);

INSERT INTO school (SCHOOL_NAME, SCHOOL_LEVEL, SCHOOL_DISTRICT_DISTRICT_ID)
VALUES ('Riverstone Academy', 'High', 4);

INSERT INTO school (SCHOOL_NAME, SCHOOL_LEVEL, SCHOOL_DISTRICT_DISTRICT_ID)
VALUES ('Stone Bridge Middle', 'Middle', 4);

INSERT INTO school (SCHOOL_NAME, SCHOOL_LEVEL, SCHOOL_DISTRICT_DISTRICT_ID)
VALUES ('Pinecrest Elementary', 'Elementary', 5);

INSERT INTO school (SCHOOL_NAME, SCHOOL_LEVEL, SCHOOL_DISTRICT_DISTRICT_ID)
VALUES ('Valley High School', 'High', 5);

COMMIT;

/*****************************
Insert into Lot table
*****************************/

INSERT INTO lot (LOT_ID, STREET, CITY, STATE_ZIP, LATITUDE, LONGITUDE, SUBDIVISION_SUB_ID)
VALUES (lot_id_seq.NEXTVAL, '123 Oak Lane', 'Oakville', 12345, 40.7128, -74.0060, 1);

INSERT INTO lot (LOT_ID, STREET, CITY, STATE_ZIP, LATITUDE, LONGITUDE, SUBDIVISION_SUB_ID)
VALUES (lot_id_seq.NEXTVAL, '456 Willow Way', 'Willowbrook', 23456, 41.8781, -87.6298, 2);

INSERT INTO lot (LOT_ID, STREET, CITY, STATE_ZIP, LATITUDE, LONGITUDE, SUBDIVISION_SUB_ID)
VALUES (lot_id_seq.NEXTVAL, '789 Meadow Street', 'Meadowville', 34567, 39.9526, -75.1652, 3);

INSERT INTO lot (LOT_ID, STREET, CITY, STATE_ZIP, LATITUDE, LONGITUDE, SUBDIVISION_SUB_ID)
VALUES (lot_id_seq.NEXTVAL, '101 Sunny Avenue', 'Sunnydale', 45678, 37.7749, -122.4194, 4);

INSERT INTO lot (LOT_ID, STREET, CITY, STATE_ZIP, LATITUDE, LONGITUDE, SUBDIVISION_SUB_ID)
VALUES (lot_id_seq.NEXTVAL, '202 River Road', 'Riverside', 56789, 42.3601, -71.0589, 5);

INSERT INTO lot (LOT_ID, STREET, CITY, STATE_ZIP, LATITUDE, LONGITUDE, SUBDIVISION_SUB_ID)
VALUES (lot_id_seq.NEXTVAL, '303 Pine Street', 'Pineville', 67890, 38.9072, -77.0369, 6);

INSERT INTO lot (LOT_ID, STREET, CITY, STATE_ZIP, LATITUDE, LONGITUDE, SUBDIVISION_SUB_ID)
VALUES (lot_id_seq.NEXTVAL, '404 Maple Drive', 'Maplewood', 78901, 33.7490, -84.3880, 7);

INSERT INTO lot (LOT_ID, STREET, CITY, STATE_ZIP, LATITUDE, LONGITUDE, SUBDIVISION_SUB_ID)
VALUES (lot_id_seq.NEXTVAL, '505 Cedar Lane', 'Oakville', 12345, 40.7282, -74.0776, 1);

INSERT INTO lot (LOT_ID, STREET, CITY, STATE_ZIP, LATITUDE, LONGITUDE, SUBDIVISION_SUB_ID)
VALUES (lot_id_seq.NEXTVAL, '606 Birch Boulevard', 'Willowbrook', 23456, 41.8852, -87.6388, 2);

INSERT INTO lot (LOT_ID, STREET, CITY, STATE_ZIP, LATITUDE, LONGITUDE, SUBDIVISION_SUB_ID)
VALUES (lot_id_seq.NEXTVAL, '707 Elm Street', 'Meadowville', 34567, 39.9585, -75.1742, 3);

COMMIT;

/*****************************
Insert into House_Style table
*****************************/

SELECT
  STYLE_NAME,
  TREAT(HOUSE_STYLE_PHOTO AS HTTPURITYPE).GETURL() AS PHOTO_URL,
  BASE_PRICE,
  STYLE_DESC,
  STYLE_SIZE,
  TREAT(FLOOR_PLAN_LINK AS HTTPURITYPE).GETURL() AS PLAN_URL
FROM House_Style;
-- Colonial, Victorian, Ranch, Craftsman, Mediterranean, Cape Cod, Tudor, Modern, Farmhouse, Bungalow
-- Inserting
INSERT INTO House_Style (style_name, house_style_photo, base_price, style_desc, floor_plan_link)
VALUES ('Colonial', HTTPURITYPE('http://example.com/colonial.jpg'), 350000, 'A classic American style with symmetrical facade and central door', HTTPURITYPE('http://example.com/colonial-plan.pdf'));

INSERT INTO House_Style (style_name, house_style_photo, base_price, style_desc, floor_plan_link)
VALUES ('Victorian', HTTPURITYPE('http://example.com/victorian.jpg'), 425000, 'Ornate architecture with intricate details and asymmetrical shape', HTTPURITYPE('http://example.com/victorian-plan.pdf'));

INSERT INTO House_Style (style_name, house_style_photo, base_price, style_desc, floor_plan_link)
VALUES ('Ranch', HTTPURITYPE('http://example.com/ranch.jpg'), 300000, 'Single-story home with open floor plan and low-pitched roof', HTTPURITYPE('http://example.com/ranch-plan.pdf'));

INSERT INTO House_Style (style_name, house_style_photo, base_price, style_desc, floor_plan_link)
VALUES ('Craftsman', HTTPURITYPE('http://example.com/craftsman.jpg'), 375000, 'Characterized by tapered columns, wide eaves, and handcrafted details', HTTPURITYPE('http://example.com/craftsman-plan.pdf'));

INSERT INTO House_Style (style_name, house_style_photo, base_price, style_desc, floor_plan_link)
VALUES ('Mediterranean', HTTPURITYPE('http://example.com/mediterranean.jpg'), 450000, 'Inspired by seaside villas with stucco walls and red tile roofs', HTTPURITYPE('http://example.com/mediterranean-plan.pdf'));

INSERT INTO House_Style (style_name, house_style_photo, base_price, style_desc, floor_plan_link)
VALUES ('Cape Cod', HTTPURITYPE('http://example.com/capecod.jpg'), 320000, 'Compact and symmetrical with steep roof and central chimney', HTTPURITYPE('http://example.com/capecod-plan.pdf'));

INSERT INTO House_Style (style_name, house_style_photo, base_price, style_desc, floor_plan_link)
VALUES ('Tudor', HTTPURITYPE('http://example.com/tudor.jpg'), 400000, 'Medieval-inspired with steep roofs, decorative half-timbering, and tall chimneys', HTTPURITYPE('http://example.com/tudor-plan.pdf'));

INSERT INTO House_Style (style_name, house_style_photo, base_price, style_desc, floor_plan_link)
VALUES ('Modern', HTTPURITYPE('http://example.com/modern.jpg'), 500000, 'Minimalist design with clean lines, large windows, and open spaces', HTTPURITYPE('http://example.com/modern-plan.pdf'));

INSERT INTO House_Style (style_name, house_style_photo, base_price, style_desc, floor_plan_link)
VALUES ('Farmhouse', HTTPURITYPE('http://example.com/farmhouse.jpg'), 380000, 'Rustic charm with wrap-around porch and practical, cozy interiors', HTTPURITYPE('http://example.com/farmhouse-plan.pdf'));

INSERT INTO House_Style (style_name, house_style_photo, base_price, style_desc, floor_plan_link)
VALUES ('Bungalow', HTTPURITYPE('http://example.com/bungalow.jpg'), 290000, 'Small, charming home with low-pitched roof and a front porch', HTTPURITYPE('http://example.com/bungalow-plan.pdf'));

COMMIT;

/*****************************
Insert into Describe table
*****************************/

INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (1, 'Mediterranean');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (6, 'Victorian');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (6, 'Bungalow');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (7, 'Farmhouse');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (8, 'Bungalow');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (8, 'Tudor');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (6, 'Modern');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (6, 'Farmhouse');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (6, 'Colonial');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (2, 'Modern');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (2, 'Cape Cod');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (1, 'Colonial');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (1, 'Victorian');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (1, 'Ranch');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (2, 'Craftsman');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (2, 'Mediterranean');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (3, 'Cape Cod');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (3, 'Tudor');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (3, 'Modern');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (4, 'Farmhouse');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (4, 'Bungalow');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (5, 'Colonial');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (5, 'Victorian');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (6, 'Ranch');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (6, 'Craftsman');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (7, 'Mediterranean');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (7, 'Cape Cod');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (8, 'Modern');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (9, 'Farmhouse');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (9, 'Bungalow');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (10, 'Colonial');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (10, 'Victorian');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (2, 'Tudor');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (3, 'Farmhouse');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (4, 'Ranch');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (5, 'Modern');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (6, 'Cape Cod');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (7, 'Bungalow');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (8, 'Craftsman');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (9, 'Colonial');
INSERT INTO describes (lot_lot_id, house_style_style_name) VALUES (10, 'Mediterranean');

COMMIT;

/*****************************
Insert into Elevation table
*****************************/

-- Colonial
INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Colonial'), 'Base design', 0, 'Colonial');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Colonial'), 'Enhanced facade with columns', 2500, 'Colonial');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Colonial'), 'Luxury trim package', 3500, 'Colonial');

-- Victorian
INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Victorian'), 'Base design', 0, 'Victorian');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Victorian'), 'Standard Victorian enhancements', 1000, 'Victorian');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Victorian'), 'Ornate gingerbread trim', 3000, 'Victorian');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Victorian'), 'Turret addition', 5000, 'Victorian');

-- Ranch
INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Ranch'), 'Base design', 0, 'Ranch');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Ranch'), 'Extended garage', 2000, 'Ranch');

-- Craftsman
INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Craftsman'), 'Base design', 0, 'Craftsman');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Craftsman'), 'Stone accent facade', 3500, 'Craftsman');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Craftsman'), 'Expanded porch', 2500, 'Craftsman');

-- Mediterranean
INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Mediterranean'), 'Base design', 0, 'Mediterranean');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Mediterranean'), 'Courtyard entry', 4000, 'Mediterranean');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Mediterranean'), 'Luxury balcony package', 3500, 'Mediterranean');

-- Bungalow
INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Bungalow'), 'Base design', 0, 'Bungalow');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Bungalow'), 'Wrap-around porch', 3000, 'Bungalow');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Bungalow'), 'Extended roofline', 2500, 'Bungalow');

-- Tudor
INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Tudor'), 'Base design', 0, 'Tudor');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Tudor'), 'Steep gabled roof', 4500, 'Tudor');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Tudor'), 'Half-timbered facade', 5000, 'Tudor');

-- Modern
INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Modern'), 'Base design', 0, 'Modern');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Modern'), 'Glass facade', 5500, 'Modern');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Modern'), 'Cantilevered roof', 6000, 'Modern');

-- Farmhouse
INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Farmhouse'), 'Base design', 0, 'Farmhouse');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Farmhouse'), 'Extended barn doors', 3500, 'Farmhouse');

INSERT INTO Elevation (elevation_id, elevation_desc, elevation_cost, house_style_style_name)
VALUES (get_next_elevation_id('Farmhouse'), 'Large wrap porch', 4000, 'Farmhouse');


-- Commit
COMMIT;

/*****************************
Insert into Room table
*****************************/

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Colonial'), 'Living Room', 280.50, 1, 'Spacious living area with fireplace', 3, 'Coffered', 'Colonial');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Colonial'), 'Kitchen', 200.75, 1, 'Large eat-in kitchen with island', 2, 'Beamed', 'Colonial');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Colonial'), 'Master Bedroom', 220.00, 2, 'Luxurious master suite with walk-in closet', 2, 'Tray', 'Colonial');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Victorian'), 'Parlor', 180.25, 1, 'Elegant parlor with ornate moldings', 2, 'Tin', 'Victorian');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Victorian'), 'Dining Room', 160.50, 1, 'Formal dining room with chandelier', 2, 'Medallion', 'Victorian');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Victorian'), 'Library', 140.75, 1, 'Cozy library with built-in bookshelves', 1, 'Coffered', 'Victorian');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Ranch'), 'Open Concept Living', 350.00, 1, 'Spacious open-plan living and dining area', 4, 'Vaulted', 'Ranch');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Ranch'), 'Master Bedroom', 200.50, 1, 'Large master bedroom with en-suite', 2, 'Flat', 'Ranch');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Ranch'), 'Family Room', 180.25, 1, 'Comfortable family room with fireplace', 3, 'Beamed', 'Ranch');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Craftsman'), 'Living Room', 240.75, 1, 'Charming living room with built-in cabinets', 3, 'Coffered', 'Craftsman');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Craftsman'), 'Dining Room', 180.00, 1, 'Dining room with wainscoting', 2, 'Beamed', 'Craftsman');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Craftsman'), 'Master Bedroom', 210.50, 2, 'Cozy master bedroom with reading nook', 2, 'Cove', 'Craftsman');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Mediterranean'), 'Great Room', 320.25, 1, 'Expansive great room with arched windows', 4, 'Vaulted', 'Mediterranean');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Mediterranean'), 'Kitchen', 230.75, 1, 'Gourmet kitchen with terracotta tiles', 2, 'Beamed', 'Mediterranean');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Mediterranean'), 'Master Suite', 280.00, 2, 'Luxurious master suite with balcony', 3, 'Tray', 'Mediterranean');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Cape Cod'), 'Living Room', 200.50, 1, 'Cozy living room with fireplace', 2, 'Flat', 'Cape Cod');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Cape Cod'), 'Kitchen', 160.25, 1, 'Charming kitchen with bay window', 2, 'Beadboard', 'Cape Cod');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Cape Cod'), 'Bedroom', 140.75, 2, 'Cozy bedroom with dormer windows', 2, 'Sloped', 'Cape Cod');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Tudor'), 'Great Hall', 300.00, 1, 'Impressive great hall with exposed beams', 3, 'Cathedral', 'Tudor');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Tudor'), 'Study', 150.50, 1, 'Wood-paneled study with fireplace', 1, 'Coffered', 'Tudor');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Tudor'), 'Master Bedroom', 220.75, 2, 'Spacious master bedroom with sitting area', 2, 'Vaulted', 'Tudor');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Modern'), 'Open Plan Living', 400.25, 1, 'Sleek open-concept living and dining area', 5, 'Flat', 'Modern');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Modern'), 'Kitchen', 250.00, 1, 'Minimalist kitchen with high-end appliances', 2, 'Dropped', 'Modern');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Modern'), 'Master Suite', 300.50, 2, 'Luxurious master suite with panoramic views', 3, 'Suspended', 'Modern');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Farmhouse'), 'Kitchen', 280.75, 1, 'Rustic kitchen with large farmhouse sink', 3, 'Beamed', 'Farmhouse');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Farmhouse'), 'Living Room', 240.25, 1, 'Cozy living room with stone fireplace', 2, 'Vaulted', 'Farmhouse');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Farmhouse'), 'Master Bedroom', 200.00, 2, 'Charming master bedroom with wood floors', 2, 'Shiplap', 'Farmhouse');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Bungalow'), 'Living Room', 180.50, 1, 'Cozy living room with built-in shelves', 2, 'Cove', 'Bungalow');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Bungalow'), 'Dining Room', 140.75, 1, 'Charming dining room with wainscoting', 1, 'Beamed', 'Bungalow');

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Bungalow'), 'Master Bedroom', 160.25, 1, 'Comfortable master bedroom with bay window', 2, 'Flat', 'Bungalow');

-- Commit
COMMIT;

/*****************************
Insert into Escrow table
*****************************/

INSERT INTO Escrow VALUES (escrow_id_seq.NEXTVAL, 1, 2000, '2160 Walmant', 'Pittsburgh', '15217');
INSERT INTO Escrow VALUES (escrow_id_seq.NEXTVAL, 3, 2500, '123 Main St', 'Philadelphia', '19103');
INSERT INTO Escrow VALUES (escrow_id_seq.NEXTVAL, 1, 3000, '456 Market St', 'New York', '10001');
INSERT INTO Escrow VALUES (escrow_id_seq.NEXTVAL, 2, 3500, '789 Broadway', 'Los Angeles', '90001');
INSERT INTO Escrow VALUES (escrow_id_seq.NEXTVAL, 2, 4000, '101 Elm St', 'Chicago', '60601');
INSERT INTO Escrow VALUES (escrow_id_seq.NEXTVAL, 3, 4500, '202 Pine St', 'Houston', '77001');
INSERT INTO Escrow VALUES (escrow_id_seq.NEXTVAL, 5, 5000, '303 Oak St', 'Phoenix', '85001');
INSERT INTO Escrow VALUES (escrow_id_seq.NEXTVAL, 2, 5500, '404 Cedar St', 'San Antonio', '78201');
INSERT INTO Escrow VALUES (escrow_id_seq.NEXTVAL, 3, 6000, '505 Maple St', 'San Diego', '92101');
INSERT INTO Escrow VALUES (escrow_id_seq.NEXTVAL, 4, 6500, '606 Birch St', 'Dallas', '75201');
COMMIT;

/*****************************
Insert into Customer table
*****************************/

INSERT INTO Customer VALUES (customer_id_seq.NEXTVAL, 'Terry', 'TEAM2', '2160 Walmant', 'Pittsburgh', '15217');
INSERT INTO Customer VALUES (customer_id_seq.NEXTVAL, 'John', 'Doe', '123 Main St', 'Philadelphia', '19103');
INSERT INTO Customer VALUES (customer_id_seq.NEXTVAL, 'Jane', 'Smith', '456 Market St', 'New York', '10001');
INSERT INTO Customer VALUES (customer_id_seq.NEXTVAL, 'Robert', 'Brown', '789 Broadway', 'Los Angeles', '90001');
INSERT INTO Customer VALUES (customer_id_seq.NEXTVAL, 'Emily', 'Davis', '101 Elm St', 'Chicago', '60601');
INSERT INTO Customer VALUES (customer_id_seq.NEXTVAL, 'Michael', 'Wilson', '202 Pine St', 'Houston', '77001');
INSERT INTO Customer VALUES (customer_id_seq.NEXTVAL, 'Sarah', 'Johnson', '303 Oak St', 'Phoenix', '85001');
INSERT INTO Customer VALUES (customer_id_seq.NEXTVAL, 'David', 'Lee', '404 Cedar St', 'San Antonio', '78201');
INSERT INTO Customer VALUES (customer_id_seq.NEXTVAL, 'Laura', 'Martinez', '505 Maple St', 'San Diego', '92101');
INSERT INTO Customer VALUES (customer_id_seq.NEXTVAL, 'James', 'Anderson', '606 Birch St', 'Dallas', '75201');

commit;

/*****************************
Insert into Bank table
*****************************/

INSERT INTO Bank VALUES (bank_id_seq.NEXTVAL, '4129987783', '4229987781', 'Walmant St', 'Pittsburgh', '15217');
INSERT INTO Bank VALUES (bank_id_seq.NEXTVAL, '4129987784', '4229987782', 'Market St', 'Philadelphia', '19103');
INSERT INTO Bank VALUES (bank_id_seq.NEXTVAL, '4129987785', '4229987783', 'Broadway', 'New York', '10001');
INSERT INTO Bank VALUES (bank_id_seq.NEXTVAL, '4129987786', '4229987784', 'Main St', 'Los Angeles', '90001');
INSERT INTO Bank VALUES (bank_id_seq.NEXTVAL, '4129987787', '4229987785', 'Elm St', 'Chicago', '60601');
INSERT INTO Bank VALUES (bank_id_seq.NEXTVAL, '4129987788', '4229987786', 'Pine St', 'Houston', '77001');
INSERT INTO Bank VALUES (bank_id_seq.NEXTVAL, '4129987789', '4229987787', 'Oak St', 'Phoenix', '85001');
INSERT INTO Bank VALUES (bank_id_seq.NEXTVAL, '4129987790', '4229987788', 'Cedar St', 'San Antonio', '78201');
INSERT INTO Bank VALUES (bank_id_seq.NEXTVAL, '4129987791', '4229987789', 'Maple St', 'San Diego', '92101');
INSERT INTO Bank VALUES (bank_id_seq.NEXTVAL, '4129987792', '4229987790', 'Birch St', 'Dallas', '75201');
commit;

/*****************************
Insert into Employee table
*****************************/

-- Inserting into Employee table with sequence for employee_id
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Tianyang', 'Chen', 'Construction_Manager');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Yi', 'Xun', 'Employee');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Taiyuan', 'Zhang', 'Sales_Representative');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Emily', 'Johnson', 'Construction_Manager');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Michael', 'Smith', 'Employee');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Sophia', 'Davis', 'Sales_Representative');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'James', 'Miller', 'Employee');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Isabella', 'Garcia', 'Construction_Manager');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Daniel', 'Wilson', 'Sales_Representative');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Olivia', 'Martinez', 'Employee');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Liam', 'Brown', 'Sales_Representative');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Emma', 'Davis', 'Construction_Manager');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Noah', 'Johnson', 'Sales_Representative');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Ava', 'Smith', 'Construction_Manager');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Oliver', 'Garcia', 'Sales_Representative');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Sophia', 'Martinez', 'Construction_Manager');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Mason', 'Hernandez', 'Sales_Representative');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Isabella', 'Lopez', 'Construction_Manager');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Lucas', 'Gonzalez', 'Sales_Representative');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Mia', 'Wilson', 'Construction_Manager');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Ethan', 'Anderson', 'Sales_Representative');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Charlotte', 'Thomas', 'Construction_Manager');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Aiden', 'Taylor', 'Sales_Representative');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Amelia', 'Moore', 'Construction_Manager');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Logan', 'Jackson', 'Sales_Representative');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Harper', 'Martin', 'Construction_Manager');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'James', 'Lee', 'Sales_Representative');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Ella', 'Perez', 'Construction_Manager');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Jackson', 'Thompson', 'Sales_Representative');
INSERT INTO Employee VALUES (employee_id_seq.NEXTVAL, 'Grace', 'White', 'Construction_Manager');

/*****************************************
Insert into Sales_Representative table
*****************************************/

-- ALTER TRIGGER ARC_FKARC_SALES_REPRESENTATIVE DISABLE;

-- Insert into Sales_Representative table using license_no_seq for license_no
INSERT INTO Sales_Representative VALUES (3, license_no_seq.NEXTVAL);
INSERT INTO Sales_Representative VALUES (6, license_no_seq.NEXTVAL);
INSERT INTO Sales_Representative VALUES (9, license_no_seq.NEXTVAL);
INSERT INTO Sales_Representative VALUES (11, license_no_seq.NEXTVAL);
INSERT INTO Sales_Representative VALUES (13, license_no_seq.NEXTVAL);
INSERT INTO Sales_Representative VALUES (15, license_no_seq.NEXTVAL);
INSERT INTO Sales_Representative VALUES (17, license_no_seq.NEXTVAL);
INSERT INTO Sales_Representative VALUES (19, license_no_seq.NEXTVAL);
INSERT INTO Sales_Representative VALUES (21, license_no_seq.NEXTVAL);
INSERT INTO Sales_Representative VALUES (23, license_no_seq.NEXTVAL);

/*****************************************
Insert into Construction_Manager table
*****************************************/

-- ALTER TRIGGER ARC_FKARC_Construction_Manager DISABLE;

-- Insert into Construction_Manager table using crew_no_seq for crew_no
INSERT INTO Construction_Manager VALUES (1, crew_no_seq.NEXTVAL);
INSERT INTO Construction_Manager VALUES (4, crew_no_seq.NEXTVAL);
INSERT INTO Construction_Manager VALUES (8, crew_no_seq.NEXTVAL);
INSERT INTO Construction_Manager VALUES (12, crew_no_seq.NEXTVAL);
INSERT INTO Construction_Manager VALUES (14, crew_no_seq.NEXTVAL);
INSERT INTO Construction_Manager VALUES (16, crew_no_seq.NEXTVAL);
INSERT INTO Construction_Manager VALUES (20, crew_no_seq.NEXTVAL);
INSERT INTO Construction_Manager VALUES (24, crew_no_seq.NEXTVAL);
INSERT INTO Construction_Manager VALUES (28, crew_no_seq.NEXTVAL);
INSERT INTO Construction_Manager VALUES (30, crew_no_seq.NEXTVAL);

COMMIT;

/*****************************************
Insert into Construction_Project table
*****************************************/

INSERT INTO Construction_Project (project_id, start_date, est_completion_date, current_stage, construction_photo, cm_employee_id)
VALUES (construction_project_id_seq.NEXTVAL, TO_DATE('09/25/2020', 'MM/DD/YYYY'), TO_DATE('09/25/2024', 'MM/DD/YYYY'), 2, HTTPURITYPE('http://example.com/xxxlink'), 1);

INSERT INTO Construction_Project (project_id, start_date, est_completion_date, current_stage, construction_photo, cm_employee_id)
VALUES (construction_project_id_seq.NEXTVAL, TO_DATE('10/01/2021', 'MM/DD/YYYY'), TO_DATE('10/01/2025', 'MM/DD/YYYY'), 5, HTTPURITYPE('http://example.com/yyylink'), 4);

INSERT INTO Construction_Project (project_id, start_date, est_completion_date, current_stage, construction_photo, cm_employee_id)
VALUES (construction_project_id_seq.NEXTVAL, TO_DATE('11/15/2019', 'MM/DD/YYYY'), TO_DATE('11/15/2023', 'MM/DD/YYYY'), 7, HTTPURITYPE('http://example.com/zzzlink'), 8);

INSERT INTO Construction_Project (project_id, start_date, est_completion_date, current_stage, construction_photo, cm_employee_id)
VALUES (construction_project_id_seq.NEXTVAL, TO_DATE('01/10/2022', 'MM/DD/YYYY'), TO_DATE('01/10/2026', 'MM/DD/YYYY'), 2, HTTPURITYPE('http://example.com/aaalink'), 12);

INSERT INTO Construction_Project (project_id, start_date, est_completion_date, current_stage, construction_photo, cm_employee_id)
VALUES (construction_project_id_seq.NEXTVAL, TO_DATE('03/20/2023', 'MM/DD/YYYY'), TO_DATE('03/20/2027', 'MM/DD/YYYY'), 6, HTTPURITYPE('http://example.com/bbblink'), 14);

INSERT INTO Construction_Project (project_id, start_date, est_completion_date, current_stage, construction_photo, cm_employee_id)
VALUES (construction_project_id_seq.NEXTVAL, TO_DATE('05/05/2020', 'MM/DD/YYYY'), TO_DATE('05/05/2024', 'MM/DD/YYYY'), 7, HTTPURITYPE('http://example.com/ccclink'), 16);

INSERT INTO Construction_Project (project_id, start_date, est_completion_date, current_stage, construction_photo, cm_employee_id)
VALUES (construction_project_id_seq.NEXTVAL, TO_DATE('07/30/2021', 'MM/DD/YYYY'), TO_DATE('07/30/2025', 'MM/DD/YYYY'), 1, HTTPURITYPE('http://example.com/dddlk'), 20);

INSERT INTO Construction_Project (project_id, start_date, est_completion_date, current_stage, construction_photo, cm_employee_id)
VALUES (construction_project_id_seq.NEXTVAL, TO_DATE('09/01/2018', 'MM/DD/YYYY'), TO_DATE('09/01/2022', 'MM/DD/YYYY'), 5, HTTPURITYPE('http://example.com/eeelink'), 24);

INSERT INTO Construction_Project (project_id, start_date, est_completion_date, current_stage, construction_photo, cm_employee_id)
VALUES (construction_project_id_seq.NEXTVAL, TO_DATE('12/12/2022', 'MM/DD/YYYY'), TO_DATE('12/12/2026', 'MM/DD/YYYY'), 7, HTTPURITYPE('http://example.com/ffflink'), 28);

commit;

/*****************************************
Insert into Task table
*****************************************/

-- Insert values into the Task table with the new sequence
INSERT INTO Task VALUES (get_next_task_id(1), 'Set the AC', 50, 1);
INSERT INTO Task VALUES (get_next_task_id(1), 'Pour the foundation', 100, 1);
INSERT INTO Task VALUES (get_next_task_id(1), 'Install plumbing', 40, 1);
INSERT INTO Task VALUES (get_next_task_id(1), 'Frame the walls', 20, 1);
INSERT INTO Task VALUES (get_next_task_id(2), 'Install roofing', 100, 2);
INSERT INTO Task VALUES (get_next_task_id(2), 'Paint the exterior', 0, 2);
INSERT INTO Task VALUES (get_next_task_id(2), 'Landscaping', 100, 2);
INSERT INTO Task VALUES (get_next_task_id(3), 'Install electrical wiring', 15, 3);
INSERT INTO Task VALUES (get_next_task_id(3), 'Finish interior drywall', 25, 3);
--INSERT INTO Task VALUES (task_id_seq.NEXTVAL, 'Install windows and doors', 35, 10);

-- Commit the transaction
COMMIT;

/*****************************************
Insert into Option_List table
*****************************************/

INSERT INTO Option_List VALUES (option_id_seq.NEXTVAL, 1, 'Modern Lighting', 600, 'Electrical', 'High-efficiency LED lighting system');
INSERT INTO Option_List VALUES (option_id_seq.NEXTVAL, 3, 'Premium Faucets', 300, 'Plumbing', 'Brushed nickel faucets for kitchen and bath');
INSERT INTO Option_List VALUES (option_id_seq.NEXTVAL, 5, 'Granite Countertops', 2, 'Interior', 'Granite countertops for kitchen and bathrooms');
INSERT INTO Option_List VALUES (option_id_seq.NEXTVAL, 4, 'Smart Thermostat', 400, 'Electrical', 'Wi-Fi enabled smart thermostat system');
INSERT INTO Option_List VALUES (option_id_seq.NEXTVAL, 6, 'Rain Showerhead', 250, 'Plumbing', 'Luxury rain showerhead for master bathroom');
INSERT INTO Option_List VALUES (option_id_seq.NEXTVAL, 2, 'Hardwood Flooring', 1500, 'Interior', 'Solid oak hardwood floors for living areas');
INSERT INTO Option_List VALUES (option_id_seq.NEXTVAL, 7, 'Recessed Lighting', 700, 'Electrical', 'LED recessed lighting throughout home');
INSERT INTO Option_List VALUES (option_id_seq.NEXTVAL, 1, 'Water Softener', 500, 'Plumbing', 'Whole-home water softener system');
INSERT INTO Option_List VALUES (option_id_seq.NEXTVAL, 3, 'Custom Cabinets', 2000, 'Interior', 'Custom kitchen cabinets with soft-close feature');
INSERT INTO Option_List VALUES (option_id_seq.NEXTVAL, 5, 'Outdoor Lighting', 350, 'Electrical', 'Outdoor LED lighting for enhanced security');

COMMIT;

/*****************************************
Insert into Decorator_Choice table
*****************************************/

INSERT INTO Decorator_Choice VALUES (dchoice_id_seq.NEXTVAL, TO_DATE('2024-09-24', 'YYYY-MM-DD'), 3, 600, 1, 1, 'Modern Lighting', 1, 'Colonial', 1);
INSERT INTO Decorator_Choice VALUES (dchoice_id_seq.NEXTVAL, TO_DATE('2024-09-25', 'YYYY-MM-DD'), 5, 300, 2, 3, 'Premium Faucets', 2, 'Victorian', 0);
INSERT INTO Decorator_Choice VALUES (dchoice_id_seq.NEXTVAL, TO_DATE('2024-09-26', 'YYYY-MM-DD'), 7, 2500, 3, 5, 'Granite Countertops', 3, 'Ranch', 2);
INSERT INTO Decorator_Choice VALUES (dchoice_id_seq.NEXTVAL, TO_DATE('2024-09-27', 'YYYY-MM-DD'), 2, 400, 4, 4, 'Smart Thermostat', 4, 'Craftsman', 1);
INSERT INTO Decorator_Choice VALUES (dchoice_id_seq.NEXTVAL, TO_DATE('2024-09-28', 'YYYY-MM-DD'), 4, 250, 5, 6, 'Rain Showerhead', 5, 'Mediterranean', 0);
INSERT INTO Decorator_Choice VALUES (dchoice_id_seq.NEXTVAL, TO_DATE('2024-09-29', 'YYYY-MM-DD'), 1, 1500, 6, 2, 'Hardwood Flooring', 6, 'Cape Cod', 2);
INSERT INTO Decorator_Choice VALUES (dchoice_id_seq.NEXTVAL, TO_DATE('2024-09-30', 'YYYY-MM-DD'), 9, 700, 7, 7, 'Recessed Lighting', 7, 'Tudor',  1);
INSERT INTO Decorator_Choice VALUES (dchoice_id_seq.NEXTVAL, TO_DATE('2024-10-01', 'YYYY-MM-DD'), 8, 500, 8, 1, 'Water Softener', 8, 'Modern', 0);
INSERT INTO Decorator_Choice VALUES (dchoice_id_seq.NEXTVAL, TO_DATE('2024-10-02', 'YYYY-MM-DD'), 6, 2000, 9, 3, 'Custom Cabinets', 9, 'Farmhouse', 2);
--INSERT INTO Decorator_Choice VALUES (dchoice_id_seq.NEXTVAL, TO_DATE('2024-10-03', 'YYYY-MM-DD'), 10, 350, 10, 5, 'Outdoor Lighting', 10, 'Bungalow', 1);

COMMIT;

/*****************************************
Insert into Sale table
*****************************************/

INSERT INTO Sale (invoice_id, date_sold, financing_method, lot_lot_id, escrow_escrow_id, cp_project_id, sr_liscense_no, customer_customer_id, bank_bank_id)
VALUES (invoice_id_seq.NEXTVAL, TO_DATE('2024-09-23', 'YYYY-MM-DD'), 'Mortgage Loan', 1, 1, 1, 1, 1, 1);

INSERT INTO Sale (invoice_id, date_sold, financing_method, lot_lot_id, escrow_escrow_id, cp_project_id, sr_liscense_no, customer_customer_id, bank_bank_id)
VALUES (invoice_id_seq.NEXTVAL, TO_DATE('2023-09-24', 'YYYY-MM-DD'), 'Seller Financing', 2, 2, 2, 2, 2, 2);

INSERT INTO Sale (invoice_id, date_sold, financing_method, lot_lot_id, escrow_escrow_id, cp_project_id, sr_liscense_no, customer_customer_id, bank_bank_id)
VALUES (invoice_id_seq.NEXTVAL, TO_DATE('2024-09-25', 'YYYY-MM-DD'), 'Mortgage Loan', 3, 3, 3, 3, 3, 3);

INSERT INTO Sale (invoice_id, date_sold, financing_method, lot_lot_id, escrow_escrow_id, cp_project_id, sr_liscense_no, customer_customer_id, bank_bank_id)
VALUES (invoice_id_seq.NEXTVAL, TO_DATE('2024-09-26', 'YYYY-MM-DD'), 'Seller Financing', 4, 4, 4, 4, 4, 4);

INSERT INTO Sale (invoice_id, date_sold, financing_method, lot_lot_id, escrow_escrow_id, cp_project_id, sr_liscense_no, customer_customer_id, bank_bank_id)
VALUES (invoice_id_seq.NEXTVAL, TO_DATE('2023-09-27', 'YYYY-MM-DD'), 'Mortgage Loan', 5, 5, 5, 5, 5, 5);

INSERT INTO Sale (invoice_id, date_sold, financing_method, lot_lot_id, escrow_escrow_id, cp_project_id, sr_liscense_no, customer_customer_id, bank_bank_id)
VALUES (invoice_id_seq.NEXTVAL, TO_DATE('2024-09-28', 'YYYY-MM-DD'), 'Seller Financing', 6, 6, 6, 6, 6, 6);

INSERT INTO Sale (invoice_id, date_sold, financing_method, lot_lot_id, escrow_escrow_id, cp_project_id, sr_liscense_no, customer_customer_id, bank_bank_id)
VALUES (invoice_id_seq.NEXTVAL, TO_DATE('2024-09-29', 'YYYY-MM-DD'), 'Mortgage Loan', 7, 7, 7, 7, 7, 7);

INSERT INTO Sale (invoice_id, date_sold, financing_method, lot_lot_id, escrow_escrow_id, cp_project_id, sr_liscense_no, customer_customer_id, bank_bank_id)
VALUES (invoice_id_seq.NEXTVAL, TO_DATE('2023-09-30', 'YYYY-MM-DD'), 'Seller Financing', 8, 8, 8, 8, 8, 8);

INSERT INTO Sale (invoice_id, date_sold, financing_method, lot_lot_id, escrow_escrow_id, cp_project_id, sr_liscense_no, customer_customer_id, bank_bank_id)
VALUES (invoice_id_seq.NEXTVAL, TO_DATE('2024-10-01', 'YYYY-MM-DD'), 'Mortgage Loan', 9, 9, 9, 9, 9, 9);

COMMIT;
/*****************************************
Insert into Chosen_style table
*****************************************/

INSERT INTO Chosen_style VALUES (schoice_id_seq.NEXTVAL, 'N', 0, 'Bungalow', 9);
INSERT INTO Chosen_style VALUES (schoice_id_seq.NEXTVAL, 'Y', 1, 'Colonial', 1);
INSERT INTO Chosen_style VALUES (schoice_id_seq.NEXTVAL, 'N', 2, 'Victorian', 6);
INSERT INTO Chosen_style VALUES (schoice_id_seq.NEXTVAL, 'Y', 0, 'Ranch', 4);
INSERT INTO Chosen_style VALUES (schoice_id_seq.NEXTVAL, 'N', 1, 'Craftsman', 2);
INSERT INTO Chosen_style VALUES (schoice_id_seq.NEXTVAL, 'Y', 2, 'Mediterranean', 7);
INSERT INTO Chosen_style VALUES (schoice_id_seq.NEXTVAL, 'Y', 1, 'Tudor', 8);
INSERT INTO Chosen_style VALUES (schoice_id_seq.NEXTVAL, 'N', 2, 'Modern', 5);
INSERT INTO Chosen_style VALUES (schoice_id_seq.NEXTVAL, 'Y', 0, 'Farmhouse', 3);

COMMIT;
