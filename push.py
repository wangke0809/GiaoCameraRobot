from dingtalkchatbot.chatbot import DingtalkChatbot


class Push(object):

    def __init__(self, token):
        self.d = DingtalkChatbot(
            'https://oapi.dingtalk.com/robot/send?access_token=%s' % token)

    def sendImage(self, title, content, url, is_at_all=False):
        self.d.send_markdown(title=title, text=content + '\n![Giao](' + url + ')\n', is_at_all=is_at_all)

    def sendMessage(self, msg, is_at_all=False):
        self.d.send_text(msg=msg, is_at_all=is_at_all)


if __name__ == '__main__':
    push = Push("a")
    push.sendImage("a", "b")
