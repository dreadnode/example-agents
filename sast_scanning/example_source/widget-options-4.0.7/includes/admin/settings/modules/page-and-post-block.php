<?php

/**
 * Widget Title Settings Module
 * Settings > Widget Options :: Hide Title
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       4.0
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Create Card Module for Hide Widget Title
 *
 * @since 4.0
 * @global $widget_options
 * @return void
 */

/*
 * Note: Please add a class "no-settings" in the <li> card if the card has no additional configuration, if there are configuration please remove the class
 */

function widgetopts_settings_page_and_post_block()
{
	global $widget_options;
	$hide_page_and_post_block 	= (isset($widget_options['settings']['hide_page_and_post_block'])) ? $widget_options['settings']['hide_page_and_post_block'] : array();
?>
	<li class="widgetopts-module-card <?php echo ($widget_options['hide_page_and_post_block'] == 'activate') ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-hide_page_and_post_block" data-module-id="hide_page_and_post_block">
		<div class="widgetopts-module-card-content">
			<h2><?php _e('Gutenberg Page & Post Block Options', 'widget-options'); ?></h2>
			<p class="widgetopts-module-desc">
				<?php _e('Extends widget options to Gutenberg blocks in pages, posts and other custom post types.', 'widget-options'); ?>
			</p>

			<div class="widgetopts-module-actions hide-if-no-js">
				<?php if ($widget_options['hide_page_and_post_block'] == 'activate') { ?>
					<button class="button button-secondary widgetopts-toggle-settings"><?php _e('Configure Settings', 'widget-options'); ?></button>
					<button class="button button-secondary widgetopts-toggle-activation"><?php _e('Disable', 'widget-options'); ?></button>
				<?php } else { ?>
					<button class="button button-secondary widgetopts-toggle-settings"><?php _e('Learn More', 'widget-options'); ?></button>
					<button class="button button-primary widgetopts-toggle-activation"><?php _e('Enable', 'widget-options'); ?></button>
				<?php } ?>

			</div>
		</div>

		<?php widgetopts_modal_start($widget_options['hide_page_and_post_block']); ?>
		<span class="dashicons widgetopts-dashicons widgetopts-no-top dashicons-admin-generic"></span>
		<h3 class="widgetopts-modal-header"><?php _e('Gutenberg Page & Post Block Options', 'widget-options'); ?></h3>
		<p>
			<?php _e("Widget Options offers a wide range of customization options for Gutenberg blocks on pages, posts, and other custom post types, enabling you more control than ever before over your site's content. Key features include alignment, roles, devices, days and dates, behaviour, logic, and animation.", 'widget-options'); ?>
		</p>
		<table class="form-table widgetopts-settings-section">
			<tr>
				<th scope="row">
					<label for="widgetopts-hide_page_and_post_block-page_and_post_block"><?php _e("Pages and Posts", 'widget-options'); ?></label>
				</th>
				<td>
					<input type="checkbox" id="widgetopts-hide_page_and_post_block-page_and_post_block" name="hide_page_and_post_block[page_and_post_block]" <?php echo widgetopts_is_checked($hide_page_and_post_block, 'page_and_post_block') ?> value="1" />
					<label for="widgetopts-hide_page_and_post_block-page_and_post_block"><?php _e('Hide on Pages and Posts Blocks', 'widget-options'); ?></label>
					<p class="description">
						<?php printf(__("Don't show widget options on pages, posts and other custom post types blocks.", 'widget-options')); ?>
					</p>
				</td>
			</tr>
		</table>
		<?php widgetopts_modal_end($widget_options['hide_page_and_post_block']); ?>

	</li>
<?php
}
add_action('widgetopts_module_cards', 'widgetopts_settings_page_and_post_block', 11);
?>