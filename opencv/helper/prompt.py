class Prompt():
    def get_user_request(question):
        user_options = {
            'y': 'Yes',
            'n': 'No',
            'q': 'Quit'
        }

        prompt = ''
        for key, value in user_options.items():
            line = '{0}: {1}\n'.format(key, value)
            prompt += line

        user_selection = input(
            '{0}\n{1}'.format(question, prompt))

        if user_options.get(user_selection) is 'Yes':
            return True
        elif user_options.get(user_selection) is 'No':
            return False
        elif user_options.get(user_selection) is 'Quit':
            quit()
        else:
            return False
