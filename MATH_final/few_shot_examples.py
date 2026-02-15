"""
Shared few-shot example pool for all math tutor implementations.

Contains 76 GSM8K-style chain-of-thought examples covering:
  - Addition, Subtraction, Multiplication, Division
  - Multi-step problems
  - Money, quantities, time, distance, fractions

At ~62 tokens each, 76 examples ≈ 4700 tokens.
This ensures truncation is needed at ALL context lengths (512, 1024, 2048, 4096).
"""

import random

ANSWER_TRIGGER = "The answer is"

# 76 few-shot examples with chain-of-thought reasoning
EXAMPLES = [
    # --- ORIGINAL 8 (from GSM8K paper) ---
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "chain": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.",
        "answer": "6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "chain": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
        "answer": "5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "chain": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
        "answer": "39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "chain": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.",
        "answer": "8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "chain": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.",
        "answer": "9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "chain": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.",
        "answer": "29"
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "chain": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.",
        "answer": "33"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "chain": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.",
        "answer": "8"
    },
    # --- ADDITION FOCUSED ---
    {
        "question": "A baker made 45 cupcakes in the morning and 32 cupcakes in the afternoon. How many cupcakes did she make in total?",
        "chain": "The baker made 45 in the morning and 32 in the afternoon. 45 + 32 = 77.",
        "answer": "77"
    },
    {
        "question": "Tom collected 28 seashells on Saturday and 35 seashells on Sunday. How many seashells does he have altogether?",
        "chain": "Tom collected 28 on Saturday and 35 on Sunday. 28 + 35 = 63.",
        "answer": "63"
    },
    {
        "question": "A library has 156 fiction books and 89 non-fiction books. How many books does the library have in total?",
        "chain": "The library has 156 fiction and 89 non-fiction books. 156 + 89 = 245.",
        "answer": "245"
    },
    {
        "question": "Maria scored 18 points in the first half and 24 points in the second half of a basketball game. How many points did she score in total?",
        "chain": "Maria scored 18 in the first half and 24 in the second half. 18 + 24 = 42.",
        "answer": "42"
    },
    # --- SUBTRACTION FOCUSED ---
    {
        "question": "A farmer had 120 apples. He sold 47 apples at the market. How many apples does he have left?",
        "chain": "The farmer started with 120 apples and sold 47. 120 - 47 = 73.",
        "answer": "73"
    },
    {
        "question": "Sarah had 85 stickers. She gave 29 stickers to her best friend. How many stickers does Sarah have now?",
        "chain": "Sarah had 85 stickers and gave away 29. 85 - 29 = 56.",
        "answer": "56"
    },
    {
        "question": "A bus had 52 passengers. At the first stop, 18 passengers got off. How many passengers are still on the bus?",
        "chain": "The bus had 52 passengers and 18 got off. 52 - 18 = 34.",
        "answer": "34"
    },
    {
        "question": "David had $250 in his savings account. He withdrew $78 to buy a gift. How much money is left in his account?",
        "chain": "David had 250 dollars and withdrew 78. 250 - 78 = 172.",
        "answer": "172"
    },
    # --- MULTIPLICATION FOCUSED ---
    {
        "question": "A teacher bought 8 packs of pencils. Each pack contains 12 pencils. How many pencils did the teacher buy in total?",
        "chain": "The teacher bought 8 packs with 12 pencils each. 8 * 12 = 96.",
        "answer": "96"
    },
    {
        "question": "A parking lot has 6 rows of cars. Each row has 15 cars. How many cars are in the parking lot?",
        "chain": "There are 6 rows with 15 cars each. 6 * 15 = 90.",
        "answer": "90"
    },
    {
        "question": "Emma reads 25 pages every day. How many pages will she read in 14 days?",
        "chain": "Emma reads 25 pages per day for 14 days. 25 * 14 = 350.",
        "answer": "350"
    },
    {
        "question": "A factory produces 340 toys per hour. How many toys does it produce in 5 hours?",
        "chain": "The factory makes 340 toys per hour for 5 hours. 340 * 5 = 1700.",
        "answer": "1700"
    },
    # --- DIVISION FOCUSED ---
    {
        "question": "There are 72 students to be divided equally into 8 groups. How many students will be in each group?",
        "chain": "72 students divided into 8 groups. 72 / 8 = 9 students per group.",
        "answer": "9"
    },
    {
        "question": "A chef has 144 cookies and wants to put them equally into 12 boxes. How many cookies go in each box?",
        "chain": "144 cookies divided into 12 boxes. 144 / 12 = 12 cookies per box.",
        "answer": "12"
    },
    {
        "question": "A rope is 96 meters long. It needs to be cut into pieces that are each 8 meters long. How many pieces can be cut?",
        "chain": "A 96 meter rope cut into 8 meter pieces. 96 / 8 = 12 pieces.",
        "answer": "12"
    },
    {
        "question": "A group of 5 friends earned $275 together. If they split the money equally, how much does each person get?",
        "chain": "275 dollars split among 5 friends. 275 / 5 = 55 dollars each.",
        "answer": "55"
    },
    # --- MULTI-STEP: ADD THEN SUBTRACT ---
    {
        "question": "A store had 200 shirts. They received a shipment of 150 new shirts, then sold 85 shirts. How many shirts does the store have now?",
        "chain": "The store started with 200 shirts, got 150 more so 200 + 150 = 350. Then sold 85, so 350 - 85 = 265.",
        "answer": "265"
    },
    {
        "question": "Lucy had 45 marbles. She won 18 marbles in a game but then lost 7. How many marbles does Lucy have?",
        "chain": "Lucy started with 45, won 18 more so 45 + 18 = 63. Then lost 7, so 63 - 7 = 56.",
        "answer": "56"
    },
    {
        "question": "A train had 312 passengers. At the first stop 45 people got on and 28 got off. How many passengers are on the train now?",
        "chain": "Started with 312. 45 got on: 312 + 45 = 357. Then 28 got off: 357 - 28 = 329.",
        "answer": "329"
    },
    {
        "question": "Ben had $500. He earned $120 from his part-time job and spent $215 on new shoes. How much money does Ben have now?",
        "chain": "Ben had 500 and earned 120 more: 500 + 120 = 620. He spent 215: 620 - 215 = 405.",
        "answer": "405"
    },
    # --- MULTI-STEP: MULTIPLY THEN ADD/SUBTRACT ---
    {
        "question": "A school bought 4 boxes of markers with 24 markers each, and also bought 15 individual markers. How many markers in total?",
        "chain": "4 boxes of 24 markers: 4 * 24 = 96 markers. Plus 15 individual: 96 + 15 = 111.",
        "answer": "111"
    },
    {
        "question": "Lisa bought 3 shirts at $18 each and a hat for $12. How much did she spend in total?",
        "chain": "3 shirts at 18 each: 3 * 18 = 54 dollars. Plus the hat: 54 + 12 = 66.",
        "answer": "66"
    },
    {
        "question": "A warehouse has 7 shelves with 36 items on each shelf. If 19 items are removed, how many items remain?",
        "chain": "7 shelves with 36 items: 7 * 36 = 252 items. Remove 19: 252 - 19 = 233.",
        "answer": "233"
    },
    {
        "question": "Jack runs 6 miles every day for a week. On Saturday he ran an extra 3 miles. How many miles did Jack run in total?",
        "chain": "Jack runs 6 miles per day for 7 days: 6 * 7 = 42 miles. Plus 3 extra: 42 + 3 = 45.",
        "answer": "45"
    },
    # --- MULTI-STEP: COMPLEX ---
    {
        "question": "A bakery sells muffins for $2 each and cookies for $1 each. If they sold 35 muffins and 48 cookies, how much money did they make?",
        "chain": "Muffins: 35 * 2 = 70 dollars. Cookies: 48 * 1 = 48 dollars. Total: 70 + 48 = 118.",
        "answer": "118"
    },
    {
        "question": "A garden has 5 rows of tomato plants with 8 plants per row and 3 rows of pepper plants with 6 plants per row. How many plants are there in total?",
        "chain": "Tomato plants: 5 * 8 = 40. Pepper plants: 3 * 6 = 18. Total: 40 + 18 = 58.",
        "answer": "58"
    },
    {
        "question": "Amy bought 4 notebooks at $3 each and 2 pens at $5 each. She paid with a $50 bill. How much change did she get?",
        "chain": "Notebooks: 4 * 3 = 12 dollars. Pens: 2 * 5 = 10 dollars. Total spent: 12 + 10 = 22. Change: 50 - 22 = 28.",
        "answer": "28"
    },
    {
        "question": "A movie theater has 15 rows with 20 seats each. If 187 seats are occupied, how many empty seats are there?",
        "chain": "Total seats: 15 * 20 = 300. Occupied: 187. Empty: 300 - 187 = 113.",
        "answer": "113"
    },
    # --- MONEY PROBLEMS ---
    {
        "question": "Jake earned $15 per hour for 8 hours of work. He spent $45 on dinner. How much money does he have left?",
        "chain": "Jake earned 15 * 8 = 120 dollars. After dinner: 120 - 45 = 75.",
        "answer": "75"
    },
    {
        "question": "A store sells apples for $2 per pound. Mrs. Smith bought 7 pounds and Mrs. Jones bought 4 pounds. How much did they spend combined?",
        "chain": "Mrs. Smith: 7 * 2 = 14 dollars. Mrs. Jones: 4 * 2 = 8 dollars. Combined: 14 + 8 = 22.",
        "answer": "22"
    },
    {
        "question": "Peter saves $35 every week. After 6 weeks, he buys a bike for $150. How much money does he have left?",
        "chain": "Peter saved 35 * 6 = 210 dollars. After buying the bike: 210 - 150 = 60.",
        "answer": "60"
    },
    {
        "question": "A family of 4 goes to a restaurant. Each meal costs $12 and they leave a $10 tip. What is the total bill?",
        "chain": "Meals: 4 * 12 = 48 dollars. Plus tip: 48 + 10 = 58.",
        "answer": "58"
    },
    # --- QUANTITY AND GROUPING ---
    {
        "question": "A school has 480 students. If each classroom holds 30 students, how many classrooms are needed?",
        "chain": "480 students divided by 30 per classroom. 480 / 30 = 16 classrooms.",
        "answer": "16"
    },
    {
        "question": "A farmer has 24 chickens and 8 cows. Each chicken lays 1 egg per day. How many eggs are laid in a week?",
        "chain": "24 chickens lay 1 egg each per day: 24 eggs per day. In a week: 24 * 7 = 168.",
        "answer": "168"
    },
    {
        "question": "A box contains 8 red balls, 12 blue balls, and 5 green balls. How many balls are in the box?",
        "chain": "Red: 8. Blue: 12. Green: 5. Total: 8 + 12 + 5 = 25.",
        "answer": "25"
    },
    {
        "question": "There are 3 classrooms. The first has 28 students, the second has 31 students, and the third has 26 students. How many students are there in all?",
        "chain": "First: 28. Second: 31. Third: 26. Total: 28 + 31 + 26 = 85.",
        "answer": "85"
    },
    # --- TIME AND RATE ---
    {
        "question": "A printer prints 12 pages per minute. How many pages can it print in 45 minutes?",
        "chain": "12 pages per minute for 45 minutes. 12 * 45 = 540.",
        "answer": "540"
    },
    {
        "question": "A car travels at 60 miles per hour. How far will it travel in 3 hours and 30 minutes?",
        "chain": "3 hours and 30 minutes is 3.5 hours. 60 * 3.5 = 210 miles.",
        "answer": "210"
    },
    {
        "question": "A factory worker assembles 8 products per hour. In a 10-hour shift, 3 products were defective and discarded. How many good products were made?",
        "chain": "Total assembled: 8 * 10 = 80. Defective removed: 80 - 3 = 77 good products.",
        "answer": "77"
    },
    {
        "question": "A pool fills at a rate of 50 gallons per hour. If the pool needs 800 gallons to be full, how many hours will it take?",
        "chain": "800 gallons at 50 gallons per hour. 800 / 50 = 16 hours.",
        "answer": "16"
    },
    # --- PERCENTAGE/FRACTION STYLE ---
    {
        "question": "A class has 40 students. Half of them are girls. One quarter of the girls play soccer. How many girls play soccer?",
        "chain": "Girls: 40 / 2 = 20. Girls who play soccer: 20 / 4 = 5.",
        "answer": "5"
    },
    {
        "question": "A pizza is cut into 8 slices. Tom ate 3 slices and his sister ate 2 slices. How many slices are left?",
        "chain": "Total slices: 8. Eaten: 3 + 2 = 5. Remaining: 8 - 5 = 3.",
        "answer": "3"
    },
    {
        "question": "A bookshelf has 60 books. One third of them are science fiction. How many science fiction books are on the shelf?",
        "chain": "One third of 60 books. 60 / 3 = 20 science fiction books.",
        "answer": "20"
    },
    {
        "question": "In a bag of 50 candies, 10 are red, 15 are blue, and the rest are green. How many green candies are there?",
        "chain": "Red and blue: 10 + 15 = 25. Green: 50 - 25 = 25.",
        "answer": "25"
    },
    # --- MORE MULTI-STEP ---
    {
        "question": "A shop owner buys 50 toys at $4 each and sells them at $7 each. What is the total profit?",
        "chain": "Cost: 50 * 4 = 200 dollars. Revenue: 50 * 7 = 350 dollars. Profit: 350 - 200 = 150.",
        "answer": "150"
    },
    {
        "question": "Kevin has 3 times as many stamps as Laura. Laura has 14 stamps. How many stamps do they have together?",
        "chain": "Kevin has 3 * 14 = 42 stamps. Together: 42 + 14 = 56.",
        "answer": "56"
    },
    {
        "question": "A rectangular garden is 12 meters long and 8 meters wide. What is the perimeter of the garden?",
        "chain": "Perimeter = 2 * (length + width) = 2 * (12 + 8) = 2 * 20 = 40 meters.",
        "answer": "40"
    },
    {
        "question": "There are 5 baskets with 9 oranges each. If 7 oranges are rotten and removed, how many good oranges remain?",
        "chain": "Total oranges: 5 * 9 = 45. After removing rotten: 45 - 7 = 38.",
        "answer": "38"
    },
    # --- ADDITIONAL EXAMPLES TO EXCEED 4096 TOKENS ---
    {
        "question": "A concert hall has 25 rows with 40 seats in each row. If 635 tickets were sold, how many seats are empty?",
        "chain": "Total seats: 25 * 40 = 1000. Tickets sold: 635. Empty seats: 1000 - 635 = 365.",
        "answer": "365"
    },
    {
        "question": "Emma earns $12 per hour babysitting. She worked 5 hours on Friday and 7 hours on Saturday. How much did she earn in total?",
        "chain": "Friday: 12 * 5 = 60 dollars. Saturday: 12 * 7 = 84 dollars. Total: 60 + 84 = 144.",
        "answer": "144"
    },
    {
        "question": "A store sells notebooks for $4 each. On Monday they sold 23 notebooks and on Tuesday they sold 31 notebooks. What was the total revenue?",
        "chain": "Monday: 23 notebooks. Tuesday: 31 notebooks. Total sold: 23 + 31 = 54. Revenue: 54 * 4 = 216 dollars.",
        "answer": "216"
    },
    {
        "question": "A swimming pool is 50 meters long. Maria swims 6 laps every day for 5 days. How many total meters does she swim?",
        "chain": "One lap is 50 meters. She swims 6 laps per day: 6 * 50 = 300 meters per day. Over 5 days: 300 * 5 = 1500 meters.",
        "answer": "1500"
    },
    {
        "question": "A fruit stand has 84 apples, 56 bananas, and 42 oranges. They sold half of each type. How many fruits are left in total?",
        "chain": "Apples left: 84 / 2 = 42. Bananas left: 56 / 2 = 28. Oranges left: 42 / 2 = 21. Total left: 42 + 28 + 21 = 91.",
        "answer": "91"
    },
    {
        "question": "A truck can carry 2400 pounds. If each box weighs 80 pounds, how many boxes can the truck carry?",
        "chain": "2400 pounds divided by 80 pounds per box. 2400 / 80 = 30 boxes.",
        "answer": "30"
    },
    {
        "question": "Mrs. Chen bought 3 dozen eggs. She used 14 eggs for baking and 8 eggs for breakfast. How many eggs does she have left?",
        "chain": "3 dozen = 3 * 12 = 36 eggs. Used: 14 + 8 = 22 eggs. Left: 36 - 22 = 14.",
        "answer": "14"
    },
    {
        "question": "A company has 180 employees. They hired 25 new people in January and 15 people resigned in February. How many employees does the company have now?",
        "chain": "Started with 180. Hired 25: 180 + 25 = 205. Resigned 15: 205 - 15 = 190.",
        "answer": "190"
    },
    {
        "question": "A train travels 85 miles per hour. How far does it travel in 4 hours?",
        "chain": "Distance = speed * time = 85 * 4 = 340 miles.",
        "answer": "340"
    },
    {
        "question": "A school cafeteria serves 320 lunches per day. How many lunches do they serve in a school week of 5 days?",
        "chain": "320 lunches per day for 5 days. 320 * 5 = 1600 lunches.",
        "answer": "1600"
    },
    {
        "question": "Ryan has 3 shelves of books. The first shelf has 18 books, the second has 24 books, and the third has 15 books. He donates 12 books. How many books does he have left?",
        "chain": "Total books: 18 + 24 + 15 = 57. After donating 12: 57 - 12 = 45.",
        "answer": "45"
    },
    {
        "question": "A recipe needs 3 cups of flour to make 24 cookies. How many cups of flour are needed to make 96 cookies?",
        "chain": "96 cookies is 96 / 24 = 4 times the recipe. So flour needed: 3 * 4 = 12 cups.",
        "answer": "12"
    },
    {
        "question": "A movie ticket costs $9 for adults and $5 for children. If a family of 2 adults and 3 children goes to the movies, how much do they pay in total?",
        "chain": "Adults: 2 * 9 = 18 dollars. Children: 3 * 5 = 15 dollars. Total: 18 + 15 = 33.",
        "answer": "33"
    },
    {
        "question": "A warehouse received 8 shipments of 125 items each. After inspection, 47 items were found defective and removed. How many good items remain?",
        "chain": "Total items: 8 * 125 = 1000. After removing defective: 1000 - 47 = 953.",
        "answer": "953"
    },
    {
        "question": "A water tank holds 500 gallons. If 35 gallons are used each day, how many gallons remain after 12 days?",
        "chain": "Used in 12 days: 35 * 12 = 420 gallons. Remaining: 500 - 420 = 80 gallons.",
        "answer": "80"
    },
    {
        "question": "A gardener plants 8 rows of flowers with 13 flowers in each row. Then she adds 6 more flowers around the border. How many flowers are there in total?",
        "chain": "Rows: 8 * 13 = 104 flowers. Plus border: 104 + 6 = 110.",
        "answer": "110"
    },
    {
        "question": "Sam collected 156 baseball cards over the summer. He gave 23 cards to his brother and traded 18 cards with a friend. How many cards does Sam have now?",
        "chain": "Started with 156. Gave away: 23 + 18 = 41. Remaining: 156 - 41 = 115.",
        "answer": "115"
    },
    {
        "question": "A carpenter has a board that is 240 centimeters long. He cuts it into 6 equal pieces. How long is each piece?",
        "chain": "240 centimeters divided into 6 pieces. 240 / 6 = 40 centimeters each.",
        "answer": "40"
    },
    {
        "question": "A candy shop sold 45 lollipops on Monday, 62 on Tuesday, and 38 on Wednesday. How many lollipops did they sell in total over the three days?",
        "chain": "Monday: 45. Tuesday: 62. Wednesday: 38. Total: 45 + 62 + 38 = 145.",
        "answer": "145"
    },
    {
        "question": "A zoo has 4 times as many birds as reptiles. If there are 15 reptiles, how many animals are there in total counting only birds and reptiles?",
        "chain": "Birds: 4 * 15 = 60. Total birds and reptiles: 60 + 15 = 75.",
        "answer": "75"
    },
]


def create_demo_text(n_shot=76, cot_flag=True, seed=42):
    """Create n-shot prompt from the shared example pool.
    
    Uses a fixed seed for reproducibility so all frameworks and all questions
    get the same ordering of few-shot examples.  This ensures the context
    manager (fixed/adaptive) always sees the same input, making framework
    comparisons fair.
    
    Args:
        n_shot: Number of examples to include (max 76)
        cot_flag: Whether to include chain-of-thought reasoning
        seed: Random seed for shuffling (fixed for reproducibility)
    
    Returns:
        Formatted demo text string
    """
    # Use a fixed seed so every call produces the same ordering
    rng = random.Random(seed)
    indices = list(range(len(EXAMPLES)))
    rng.shuffle(indices)
    
    demo_text = ""
    for i in indices[:n_shot]:
        ex = EXAMPLES[i]
        if cot_flag:
            demo_text += f"Q: {ex['question']}\nA: {ex['chain']} {ANSWER_TRIGGER} {ex['answer']}.\n\n"
        else:
            demo_text += f"Q: {ex['question']}\nA: {ANSWER_TRIGGER} {ex['answer']}.\n\n"
    return demo_text


def get_few_shot_examples():
    """Return the full example pool"""
    return EXAMPLES
