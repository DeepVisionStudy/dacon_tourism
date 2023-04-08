import wandb

api_key = ''

def wandb_init(args):
    wandb.login(key=api_key)
    wandb.init(
        project='dacon_tourism',
        entity='',
        name=args.work_dir_exp.split('\\')[-1],
        tags=[args.text_model, args.image_model],
        reinit=True,
        config=args.__dict__,
    )