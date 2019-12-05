from dingtalkchatbot.chatbot import DingtalkChatbot
import sys
sys.path.append("../")
import config

webhook = 'https://oapi.dingtalk.com/robot/send?access_token=' + config.dingTalkWebhookAccessToken
xiaoding = DingtalkChatbot(webhook)
# Text MSG and at all
# xiaoding.send_text(msg='Giao!看我头像')

xiaoding.send_markdown(title='x', text='![Giao](http://xxx/20191205/11:08:13.jpg)\n',
                           is_at_all=True)