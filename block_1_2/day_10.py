class Students:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def introduce(self):
        print(f'Я {self.name}, мені {self.age} років, моя оцінка - {self.grade}')

students = [Students('Den', 20, 90),
            Students('Gila', 18, 91),
            Students('Bob', 19, 85),
            Students('Lilia', 21, 87),
            Students('Roman', 19, 95),]

for student in students:
    student.introduce()

for student in students:
   if student.grade >= 90:
       student.introduce()

for student in students:
    if student.name == 'Gila':
        student.introduce()

def counting():
    total = 0

    for student in students:
     total += student.grade

    return total / len(students)

bal = counting()

print(round(bal))